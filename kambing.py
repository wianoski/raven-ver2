import sys
import os
#import RPi.GPIO as GPIO
#GPIO.setup(16,GPIO.OUT)
#GPIO.setup(20,GPIO.IN)
sys.path.insert(1, os.getcwd())

import time
import math
from pymavlink import mavutil
from dronekit import connect, VehicleMode, LocationGlobal
import balloon_config
from balloon_video import balloon_video
from balloon_utils import get_distance_from_pixels, wrap_PI, shift_to_origin
from position_vector import PositionVector
from find_balloon import balloon_finder
from fake_balloon import balloon_sim
import pid
from fire import fire_finder
from fireMerah import merah_finder
from attitude_history import AttitudeHistory
from CircleDetector import CircleDetector
from sc_logger import sc_logger

"""
Misi takeoff-wp-rtl
Jika next command terdeteksi, dia ubah ke guided-cek api-auto.
Jika sudah terdeteksi 3 titik api, maka MISI dinyatakan SELESAI.
Jika group titik api sudah terdeteksi salah satunya, maka waypoint selanjutnya diLEWATI.

This is an early guess at a top level controller that uses the DroneAPI and OpenCV magic
to try and win the challenge.

General approach:
* A state machine will need to know if we are in the portion of the route where we should even
be looking for balloons (by looking at vehicle mode == AUTO and between wpt #x and #y?)
* Periodically grab openCV frames and ask the image analysis function if it thinks there
is a balloon there and if so - what direction should we go.
* Initially we use current vehicle position + orientation + camera orentation to do a GUIDED
goto for a wpt beyond where the vehicle should go.
* If our altitude gets too low, or we get too far from the 'bee-line' path, abandon looking for
this particular balloon (to prevent crashing or wasting too much time looking for balloons)
* Until vehicle enters auto mode the strategy will only look for balloons (and generate
log messages
* If there was a mavlink message for telling the vehicle to shoot for a particular heading/altitude
rather than just going to wpts that might be a good optimization

To run this module:
* Run mavproxy.py with the correct options to connect to your vehicle
* module load api
* api start balloon-strategy.py

or

* execute linux_run_strategy.sh

Then arm it an takeoff as in mavproxy_sequence.txt
"""

class BalloonStrategy(object):
    def __init__(self):

        # connect to vehicle with dronekit
        self.vehicle = self.get_vehicle_with_dronekit()
        # mode 0 = balloon finder ; mode 1 = Precision Land & Landing on moving platform ; mode 2 = detection & following mode
        # mode 3 = Precision Land without timout ; mode 4 = Fire Detetction
        self.mission_on_guide_mode = balloon_config.config.get_integer('general','mode_on_guide',4)

        self.camera_width = balloon_config.config.get_integer('camera','width',640)
        self.camera_height = balloon_config.config.get_integer('camera','height',480)
        self.camera_hfov = balloon_config.config.get_float('camera', 'horizontal-fov', 72.42)
        self.camera_vfov = balloon_config.config.get_float('camera', 'vertical-fov', 43.3)

        # initialised flag
        self.home_initialised = False
        # timer to intermittently check for home position
        self.last_home_check = time.time()

        # historical attitude
        self.att_hist = AttitudeHistory(self.vehicle, 2.0)
        self.attitude_delay = 0.0               # expected delay between image and attitude

        # search variables
        self.search_state = 0                   # 0 = not search, 1 = spinning and looking, 2 = rotating to best balloon and double checking, 3 = finalise yaw 
        self.search_start_heading = None        # initial heading of vehicle when search began
        self.search_target_heading = None       # the vehicle's current target heading (updated as vehicle spins during search)
        self.search_heading_change = None       # heading change (in radians) performed so far during search
        self.search_balloon_pos = None          # position (as an offset from home) of closest balloon (so far) during search
        self.search_balloon_heading = None      # earth-frame heading (in radians) from vehicle to closest balloon
        self.search_balloon_pitch_top = None        # earth-frame pitch (in radians) from vehicle to closest balloon
        self.search_balloon_distance = None     # distance (in meters) from vehicle to closest balloon
        self.targeting_start_time = 0           # time vehicle first pointed towards final target (used with delay_time below)
        self.targeting_delay_time = balloon_config.config.get_float('general','SEARCH_TARGET_DELAY',2.0)    # time vehicle waits after pointing towards ballon and before heading towards it
        self.search_yaw_speed = balloon_config.config.get_float('general','SEARCH_YAW_SPEED',5.0) 

        # vehicle mission
        self.mission_cmds = None
        self.mission_alt_min = 0                # min altitude from NAV_GUIDED mission command (we ignore balloons below this altitude).  "0" means no limit
        self.mission_alt_max = 0                # max altitude from NAV_GUIDED mission command (we ignore balloons above this altitude).  "0" means no limit
        self.mission_distance_max = 0           # max distance from NAV_GUIDED mission command (we ignore balloons further than this distance).  "0" means no limit

        # we are not in control of vehicle
        self.controlling_vehicle = False

        # vehicle position captured at time camera image was captured
        self.vehicle_pos = None

        # balloon direction and position estimate from latest call to analyse_image
        self.balloon_found = False
        self.fire_found = False
        self.balloon_pitch = None
        self.balloon_pitch_top = None           # earth-frame pitch (in radians) from vehicle to top of closest balloon
        self.balloon_heading = None
        self.balloon_distance = None
        self.balloon_pos = None             # last estimated position as an offset from home

        # time of the last target update sent to the flight controller
        self.guided_last_update = time.time()

        # latest velocity target sent to flight controller
        self.guided_target_vel = None

        # time the target balloon was last spotted
        self.last_spotted_time = 0

        # if we lose sight of a balloon for this many seconds we will consider it lost and give up on the search
        self.lost_sight_timeout = 15

        # The module only prints log messages unless the vehicle is in GUIDED mode (for testing).
        # Once this seems to work reasonablly well change self.debug to False and then it will
        # actually _enter_ guided mode when it thinks it sees a balloon
        self.debug = balloon_config.config.get_boolean('general','debug',True)

        # use the simulator to generate fake balloon images
        self.use_simulator = balloon_config.config.get_boolean('general','simulate',False)

        # start background image grabber
        if not self.use_simulator:
            balloon_video.start_background_capture()

        # initialise video writer
        self.writer = None

        # horizontal velocity pid controller.  maximum effect is 10 degree lean
        xy_p = balloon_config.config.get_float('general','VEL_XY_P',1.0)
        xy_i = balloon_config.config.get_float('general','VEL_XY_I',0.0)
        xy_d = balloon_config.config.get_float('general','VEL_XY_D',0.0)
        xy_imax = balloon_config.config.get_float('general','VEL_XY_IMAX',10.0)
        self.vel_xy_pid = pid.pid(xy_p, xy_i, xy_d, math.radians(xy_imax))

        # vertical velocity pid controller.  maximum effect is 10 degree lean
        z_p = balloon_config.config.get_float('general','VEL_Z_P',2.5)
        z_i = balloon_config.config.get_float('general','VEL_Z_I',0.0)
        z_d = balloon_config.config.get_float('general','VEL_Z_D',0.0)
        z_imax = balloon_config.config.get_float('general','VEL_IMAX',10.0)
        self.vel_z_pid = pid.pid(z_p, z_i, z_d, math.radians(z_imax))

        # velocity controller min and max speed
        self.vel_speed_min = balloon_config.config.get_float('general','VEL_SPEED_MIN',1.0)
        self.vel_speed_max = balloon_config.config.get_float('general','VEL_SPEED_MAX',4.0)
        self.vel_speed_last = 0.0   # last recorded speed
        self.vel_accel = balloon_config.config.get_float('general','VEL_ACCEL', 0.5)    # maximum acceleration in m/s/s
        self.vel_dist_ratio = balloon_config.config.get_float('general','VEL_DIST_RATIO', 0.5) 

        # pitch angle to hit balloon at.  negative means come down from above
        self.vel_pitch_target = math.radians(balloon_config.config.get_float('general','VEL_PITCH_TARGET',-5.0))

        # velocity controller update rate
        self.vel_update_rate = balloon_config.config.get_float('general','VEL_UPDATE_RATE_SEC',0.2)

        # stats
        self.num_frames_analysed = 0
        self.stats_start_time = 0
        #self.attitude = (0,0,0)


        #FROM PRECISION LAND!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #how many times to attempt a land before giving up
        self.search_attempts = balloon_config.config.get_integer('general','search_attempts', 5)

        #The timeout between losing the target and starting a climb/scan
        self.settle_time = balloon_config.config.get_integer('general','settle_time', 1.5)

        #how high to climb in meters to complete a scan routine
        self.climb_altitude = balloon_config.config.get_integer('general','climb_altitude', 20)

        #the max horizontal speed sent to autopilot
        #self.vel_speed_max = balloon_config.config.get_float('general', 'vel_speed_max', 5)

        #P term of the horizontal distance to velocity controller
        self.dist_to_vel = balloon_config.config.get_float('general', 'dist_to_vel', 0.15)

        #Descent velocity
        self.descent_rate = balloon_config.config.get_float('general','descent_rate', 0.5)

        #roll/pitch valupixee that is considered stable
        self.stable_attitude = balloon_config.config.get_float('general', 'stable_attitude', 0.18)

        #Climb rate when executing a search
        self.climb_rate = balloon_config.config.get_float('general','climb_rate', -2.0)

        #The height at a climb is started if no target is detected
        self.abort_height = balloon_config.config.get_integer('general', 'abort_height', 10)

        #when we have lock on target, only descend if within this radius
        self.descent_radius = balloon_config.config.get_float('general', 'descent_radius', 1.0)

        #The height at which we lock the position on xy axis, default = 1
        self.landing_area_min_alt = balloon_config.config.get_integer('general', 'landing_area_min_alt', 0)

        #The radius of the cylinder surrounding the landing pad
        self.landing_area_radius = balloon_config.config.get_integer('general', 'landing_area_radius', 20)

        #Whether the landing program can be reset after it is disabled
        self.allow_reset = balloon_config.config.get_boolean('general', 'allow_reset', True)

        #Run the program no matter what mode or location; Useful for debug purposes
        self.always_run = balloon_config.config.get_boolean('general', 'always_run', True)

        #whether the companion computer has control of the autopilot or not
        self.in_control = False

        #The timeout between losing the target and starting a climb/scan
        self.settle_time = balloon_config.config.get_integer('general','settle_time', 1.5)
        
        #how many frames have been captured
        self.frame_count = 0
        
        #Reset state machine
        self.initialize_landing()

        # Variable sendiri
        self.lanjut_cmd = 1
        self.arm = True
        self.a = False
        self.b = 0
        self.c = False
        self.relay(0)  #servo
        self.relay1(0) #indikator
        self.relay2(0) #magnet
        self.servo(0)
        self.waktu = 0
        self.waktu1 = 0
        self.drop = 0
        #GPIO.output(16,GPIO.HIGH)
        #GPIO.output(20,GPIO.LOW)
    
    # connect to vehicle with dronekit
    def get_vehicle_with_dronekit(self):
        connection_str = balloon_config.config.get_string('dronekit','connection_string','/dev/ttyUSB0') 
        connection_baud = balloon_config.config.get_integer('dronekit','baud',921600)
        #connection_baud = balloon_config.config.get_integer('dronekit','baud',1500000)
        print "connecting to vehicle on %s, baud=%d" % (connection_str, connection_baud)
        return connect(connection_str, baud=connection_baud)
        print "connected to vehicle"

    # fetch_mission - fetch mission from flight controller
    def fetch_mission(self):
        # download the vehicle waypoints
        print "fetching mission.."
        self.mission_cmds = self.vehicle.commands
        self.mission_cmds.download()
        self.mission_cmds.wait_ready()
        if not self.mission_cmds is None:
            print "retrieved mission with %d commands" % self.mission_cmds.count
        else:
            print "failed to retrieve mission"

    # check home - intermittently checks for changes to the home location
    def check_home(self):

        # return immediately if home has already been initialised
        if self.home_initialised:
            return True

        # check for home no more than once every two seconds
        if (time.time() - self.last_home_check > 2):

            # update that we have performed a status check
            self.last_home_check = time.time()

            # check if we have a vehicle
            if self.vehicle is None:
                self.vehicle = self.get_vehicle_with_dronekit()
                return

            # download the vehicle waypoints (and home) if we don't have them already
            if self.mission_cmds is None:
                self.fetch_mission()
                return False

            # ensure the vehicle's position is known
            if self.vehicle.location.global_relative_frame is None:
                print "waiting for vehicle position.."
                return False
            if self.vehicle.location.global_relative_frame.lat is None or self.vehicle.location.global_relative_frame.lon is None or self.vehicle.location.global_relative_frame.alt is None:
                print "waiting for vehicle position.."
                return False

            # get home location from mission command list
            if self.vehicle.home_location is None:
                print "waiting for home location.."
                self.fetch_mission()
                return False

            # sanity check home position
            if self.vehicle.home_location.lat == 0 and self.vehicle.home_location.lon == 0:
                print "home position all zero, re-fetching.."
                self.fetch_mission()
                return False

            # set home position
            PositionVector.set_home_location(LocationGlobal(self.vehicle.home_location.lat,self.vehicle.home_location.lon,0))
            self.home_initialised = True
            print "got home location"

            # To-Do: if we wish to have the same home position as the flight controller
            # we must download the home waypoint again whenever the vehicle is armed

        # return whether home has been initialised or not
        return self.home_initialised

    # checks if video output should be started
    def check_video_out(self):

        # return immediately if video has already been started
        if not self.writer is None:
            return
        #self.writer = balloon_video.open_video_writer()
        # start video once vehicle is armed
        if self.vehicle.armed and self.arm:
            sc_logger.text(sc_logger.GENERAL, "armed")
            self.arm = False
            #self.writer = balloon_video.open_video_writer()
        if not self.vehicle.armed and not self.arm:
            sc_logger.text(sc_logger.GENERAL, "disarm")
            self.arm = True

    # buat trigger payload, disini pake relay
    def servo(self, sv):
         #input the message
         msg = self.vehicle.message_factory.command_long_encode(0, 0, # target system, target component
                                                     mavutil.mavlink.MAV_CMD_DO_SET_RELAY,
                                                     0, # konfirmasi
                                                     3, # pin relay pada AUX OUT 3
                                                     sv, # 1 = ON, 0 = OFF
                                                     0, 0, 0, 0, 0) # param 1 ~ 5 ga dipake
         # send command to vehicle
         self.vehicle.send_mavlink(msg)
         self.vehicle.flush()
		 
    # buat trigger payload, disini pake relay
    def relay(self, st):
         #input the message
         msg = self.vehicle.message_factory.command_long_encode(0, 0, # target system, target component
                                                     mavutil.mavlink.MAV_CMD_DO_SET_RELAY,
                                                     0, # konfirmasi
                                                     0, # pin 0 relay pada AUX OUT 5
                                                     st, # 1 = ON, 0 = OFF
                                                     0, 0, 0, 0, 0) # param 1 ~ 5 ga dipake
         # send command to vehicle
         self.vehicle.send_mavlink(msg)
         self.vehicle.flush()

    # buat trigger payload, disini pake relay
    def relay1(self, st1):
         #input the message
         msg = self.vehicle.message_factory.command_long_encode(0, 0, # target system, target component
                                                     mavutil.mavlink.MAV_CMD_DO_SET_RELAY,
                                                     0, # konfirmasi
                                                     1, # pin 1 relay pada AUX OUT 6
                                                     st1, # 1 = ON, 0 = OFF
                                                     0, 0, 0, 0, 0) # param 1 ~ 5 ga dipake
         # send command to vehicle
         self.vehicle.send_mavlink(msg)
         self.vehicle.flush()

    # buat trigger payload, disini pake relay
    def relay2(self, st2):
         #input the message
         msg = self.vehicle.message_factory.command_long_encode(0, 0, # target system, target component
                                                     mavutil.mavlink.MAV_CMD_DO_SET_RELAY,
                                                     0, # konfirmasi
                                                     2, # pin 2 relay pada AUX OUT 4
                                                     st2, # 1 = ON, 0 = OFF
                                                     0, 0, 0, 0, 0) # param 1 ~ 5 ga dipake
         # send command to vehicle
         self.vehicle.send_mavlink(msg)
         self.vehicle.flush()
    
    # check_status - poles vehicle' status to determine if we are in control of vehicle or not
    def check_status(self):

        # Reset lanjut_cmd
        if (self.vehicle.mode.name == "LOITER" or self.vehicle.mode.name == "STABILIZE") and self.lanjut_cmd > 1:
            sc_logger.text(sc_logger.GENERAL, "Resetting Mission")
            self.c = False
            self.drop = 0
            self.relay(0)
            self.lanjut_cmd = 1

        # download the vehicle waypoints if we don't have them already
        # To-Do: do not load waypoints if vehicle is armed
        if self.mission_cmds is None:
            self.fetch_mission()
            return

        # Check for Auto mode and executing Nav-Guided command
        if self.vehicle.mode.name == "AUTO":
            self.relay1(0) #lampu indikator off
            # get active command number
            active_command = self.vehicle.commands.next
            #print "Active Command dan Lanjut Command adalah %s, %s" % (active_command,self.lanjut_cmd)
            sc_logger.text(sc_logger.GENERAL, "Active dan Lanjut Command: %s, %s" % (active_command, self.lanjut_cmd))

            # Buat ubah ke GUIDED
            if active_command == 1 and self.lanjut_cmd == 1:
                sc_logger.text(sc_logger.GENERAL, "Taking Off!!!")
                self.lanjut_cmd = 3
            if (active_command < self.vehicle.commands.count) and (active_command == self.lanjut_cmd): # WP cek api
                self.lanjut_cmd += 1
                sc_logger.text(sc_logger.GENERAL, "ON TARGET, ubah Mode ke GUIDED")
                self.c = True
                self.a = True
                self.waktu1 = 0
                self.vehicle.mode = VehicleMode("GUIDED")
                self.vehicle.flush()
            if active_command == self.vehicle.commands.count:
                sc_logger.text(sc_logger.GENERAL, "Misi selesai, Mode RTL dan LAND")
                self.vehicle.mode = VehicleMode("RTL")
                self.vehicle.flush()

        # we are active in guided mode
        if self.vehicle.mode.name == "GUIDED":
            if not self.controlling_vehicle:
                self.controlling_vehicle = True
                print "taking control of vehicle in GUIDED"
                print "Mulai cari Target"
                sc_logger.text(sc_logger.GENERAL, "taking control of vehicle in GUIDED dan mulai cari Target")
            return
    
        # if we got here then we are not in control
        if self.controlling_vehicle:
            self.controlling_vehicle = False
            print "giving up control of vehicle in %s" % self.vehicle.mode.name 

    # condition_yaw - send condition_yaw mavlink command to vehicle so it points at specified heading (in degrees)
    def condition_yaw(self, heading):
        # create the CONDITION_YAW command
        msg = self.vehicle.message_factory.mission_item_encode(0, 0,  # target system, target component
                                                     0,     # sequence
                                                     mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, # frame
                                                     mavutil.mavlink.MAV_CMD_CONDITION_YAW,         # command
                                                     2, # current - set to 2 to make it a guided command
                                                     0, # auto continue
                                                     heading, 0, 0, 0, 0, 0, 0) # param 1 ~ 7
        # send command to vehicle
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    # send_nav_velocity - send nav_velocity command to vehicle to request it fly in specified direction
    def send_nav_velocity(self, velocity_x, velocity_y, velocity_z):
        # create the SET_POSITION_TARGET_LOCAL_NED command
        msg = self.vehicle.message_factory.set_position_target_local_ned_encode(
                                                     0,       # time_boot_ms (not used)
                                                     0, 0,    # target system, target component
                                                     mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
                                                     0b0000111111000111, # type_mask (only speeds enabled)
                                                     0, 0, 0, # x, y, z positions (not used)
                                                     velocity_x, velocity_y, velocity_z, # x, y, z velocity in m/s
                                                     0, 0, 0, # x, y, z acceleration (not used)
                                                     0, 0)    # yaw, yaw_rate (not used) 
        # send command to vehicle
        self.vehicle.send_mavlink(msg)
        self.vehicle.flush()

    # advance_current_cmd - ask vehicle to advance to next command (i.e. abort current command)
    def advance_current_cmd(self):

        # exit immediately if we are not in AUTO mode or not controlling the vehicle
        if not self.vehicle.mode.name == "AUTO" or not self.controlling_vehicle:
            return

        # download the vehicle waypoints if we don't have them already
        if self.mission_cmds is None:
            self.fetch_mission()

        # get active command
        active_command = self.vehicle.commands.next

        # ensure there is one more command at least
        if (self.vehicle.commands.count > active_command):
            # create the MISSION_SET_CURRENT command
            msg = self.vehicle.message_factory.mission_set_current_encode(0, 0, active_command+1) # target system, target component, sequence
            # send command to vehicle
            self.vehicle.send_mavlink(msg)
            self.vehicle.flush()
        else:
            print "Failed to advance command"

    # get_frame - get a single frame from the camera or simulator
    def get_frame(self):
        if self.use_simulator:
            veh_pos = PositionVector.get_from_location(self.vehicle.location.global_relative_frame)
            frame = balloon_sim.get_simulated_frame(veh_pos, self.vehicle.attitude.roll, self.vehicle.attitude.pitch, self.vehicle.attitude.yaw)
        else:
            frame = balloon_video.get_image()
        return frame

    # get image from balloon_video class and look for balloon, results are held in the following variables:
    #    self.balloon_found : set to True if balloon is found, False otherwise
    #    self.balloon_pitch : earth frame pitch (in radians) from vehicle to balloon (i.e. takes account of vehicle attitude)
    #    self.balloon_heading : earth frame heading (in radians) from vehicle to balloon
    #    self.balloon_distance : distance (in meters) from vehicle to balloon
    #    self.balloon_pos : position vector of balloon's position as an offset from home in meters

    def output_frame_rate_stats(self):
        # get current time
        now = time.time()

        # output stats each 10seconds
        time_diff = now - self.stats_start_time 
        if time_diff < 10 or time_diff <= 0:
            return

        # output frame rate
        frame_rate = self.num_frames_analysed / time_diff
        print "FrameRate: %f (%d frames in %f seconds)" % (frame_rate, self.num_frames_analysed, time_diff)

        # reset stats
        self.num_frames_analysed = 0
        self.stats_start_time = now


    

    ### FROM PRECSION LAND !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    def analisis_image(self):
        # record time
        now = time.time()
        #PositionVectors = PositionVector()
        # capture vehicle position
        self.vehicle_pos = PositionVector.get_from_location(self.vehicle.location.global_relative_frame)

        # capture vehicle attitude in buffer
        self.att_hist.update()

        # get delayed attitude from buffer
        veh_att_delayed = self.att_hist.get_attitude(now - self.attitude_delay)

        # get new image from camera
        f = self.get_frame()
        #create an image processor
        if self.mission_on_guide_mode == 4:
            if self.vehicle.commands.next == 3 or self.vehicle.commands.next == 4 or self.vehicle.commands.next == 7 or self.vehicle.commands.next == 8 :
                self.fire_found, xpos, ypos, size = merah_finder.filter(f)
                #result = detector.analyze_frame(f)
                self.attitude = self.get_attitude()
                if self.fire_found == True:
                    self.b = 0
                    sc_logger.text(sc_logger.GENERAL, "DROPPING MP : Target terdeteksi!")
                    print xpos,ypos,size
                    if self.vehicle.mode.name == "GUIDED":
                        sc_logger.text(sc_logger.GENERAL, "DROPPING MP : Target terdeteksi, mendekati target!")
                    self.move_to_target_fire(xpos,ypos,size,self.attitude,self.vehicle_pos)
                    self.relay1(1) #lampu indikator on
                    #self.servo(0)#test untuk servo kamera
                elif self.vehicle.mode.name == "GUIDED":
                    self.relay1(0) #lampu indikator off
                    #self.servo(1)#test untuk servo kamera
                    self.b += 1
                    if self.b > 25: #15
                        self.b = 0
                        #print "Api tidak terdeteksi, ubah mode ke AUTO"
                        #self.lanjut_cmd += 1
                        sc_logger.text(sc_logger.GENERAL, "Target tidak terdeteksi, ubah mode ke AUTO")
                        self.a = False
                        self.waktu1 = 0
                        #self.c = True
                        self.controlling_vehicle = False
                        self.vehicle.mode = VehicleMode("AUTO")
                        self.vehicle.flush()                
            else :
                self.fire_found, xpos, ypos, size = fire_finder.filter(f)
                #result = detector.analyze_frame(f)
                self.attitude = self.get_attitude()
                if self.fire_found == True:
                    self.b = 0
                    sc_logger.text(sc_logger.GENERAL, "PICK or DROP LOAD : Target terdeteksi!")
                    print xpos,ypos,size
                    if self.vehicle.mode.name == "GUIDED":
                        sc_logger.text(sc_logger.GENERAL, "PICK or DROP LOAD :Target terdeteksi, mendekati target!")
                    self.move_to_target_fire(xpos,ypos,size,self.attitude,self.vehicle_pos)
                    self.relay1(1) #lampu indikator on
                    #self.servo(0)#test untuk servo kamera
                elif self.vehicle.mode.name == "GUIDED":
                    self.relay1(0) #lampu indikator off
                    #self.servo(1)#test untuk servo kamera
                    self.b += 1
                    if self.b > 25: #15
                        self.b = 0
                        #print "Api tidak terdeteksi, ubah mode ke AUTO"
                        #self.lanjut_cmd += 1
                        sc_logger.text(sc_logger.GENERAL, "Target tidak terdeteksi, ubah mode ke AUTO")
                        self.a = False
                        self.waktu1 = 0
                        #self.c = True
                        self.controlling_vehicle = False
                        self.vehicle.mode = VehicleMode("AUTO")
                        self.vehicle.flush()
            #rend_Image = detector.add_target_highlights(f, result[3])
            sc_logger.image(sc_logger.GUI, f)

            if not self.writer is None:
            # save image for debugging later
                self.writer.write(f)

            # increment frames analysed for stats
            self.num_frames_analysed += 1
            if self.stats_start_time == 0:
                self.stats_start_time = time.time()

    #get_attitude - returns pitch, roll, and yaw of vehicle
    def get_attitude(self):
        return self.vehicle.attitude

    #initialize_landing - reset the state machine which controls the flow of the landing routine
    def initialize_landing(self):
        
        #how mant times we have attempted landing
        self.attempts = 0

        #Last time in millis since we had a valid target
        self.last_valid_target = 0

        #State variable climbing to scan for the target
        self.climbing = False

        #State variable which determines if this program will continue to run
        self.pl_enabled = True

        #State variable used to represent if autopilot is active
        self.initial_descent = True

        #State variable which represents a know target in landing area
        self.target_detected = False

    #move_to_target - fly aircraft to landing pad
    def move_to_target_fire(self,x,y,size,attitude,location):

        # exit immediately if we are not controlling the vehicle
        if not self.controlling_vehicle:
            return

        # get active command
        active_command = self.vehicle.commands.next

        # get current time
        now = time.time()

        # exit immediately if it's been too soon since the last update
        if (now - self.guided_last_update) < self.vel_update_rate:
            return;

        # if we have a new balloon position recalculate velocity vector
        if (self.fire_found):
            #x,y = target_info[1]
            
            #pitch_dir,yaw_dir = balloon_finder.pixels_to_direction(x, y, attitude.roll, attitude.pitch, attitude.yaw)
            #yaw_dir = yaw_dir % 360 
            #print "Target Yaw:%f" %(yaw_dir)
            #target_distance = target_info[2]
            #print "Target Distance:%f" %(target_distance*100)
            #shift origin to center of image

            x,y = shift_to_origin((x,y),self.camera_width,self.camera_height)
            hfov = self.camera_hfov
            vfov = self.camera_vfov

            #stabilize image with vehicle attitude
            x -= (self.camera_width / hfov) * math.degrees(attitude.roll)
            y += (self.camera_height / vfov) * math.degrees(attitude.pitch)


            #convert to distance
            X, Y = self.pixel_point_to_position_xy((x,y),location.z)

            #convert to world coordinates
            target_headings = math.atan2(Y,X) #% (2*math.pi)
            target_heading = (attitude.yaw - target_headings) 
            target_distance = math.sqrt(X**2 + Y**2)
         
            #sc_logger.text(sc_logger.GENERAL, "Distance to target: {0}".format(round(target_distance,2)))
        
            
            #calculate speed toward target
            speed = target_distance * self.dist_to_vel
            #apply max speed limit
            speed = min(speed,self.vel_speed_max)

            #caalculate cartisian speed
            vx = speed * math.sin(target_heading) * -1.0
            vy = speed * math.cos(target_heading) #*-1.0

            print "Found Target go to vx:%f vy:%f Alt:%f target_distance:%f headings:%f heading:%f " % (vx,vy,location.z,target_distance,target_headings,target_heading)

            #only descend when on top of target
            if(location.z > 3.5):
                vz = 0.25
            else:
                vz = 0
					
            if active_command == 3: #PICK MP1
                #Jika ketinggian sudah dibawah 4 meter, maka MAGNET ON
                #self.servo(0)#kamera siap untuk pick
                if (location.z < 4.0):
                    if(location.z > 1.5):
                        vz = 0.15
                    else:
                        vz = 0
                        if (self.c == True):
                            self.relay2(1) #Capit ON 
                            self.waktu = 0
                            self.c = False
                        print ("Payload sudah diambil, lanjut misi")
                        sc_logger.text(sc_logger.GENERAL, "Payload sudah diambil, lanjut misi")
                        self.controlling_vehicle = False
                        self.vehicle.mode = VehicleMode("AUTO")
                        self.vehicle.flush()

            # if active_command = 3:
            #     if
						
            if active_command == 4: #DROP MP1
                #self.servo(1)#kamera siap untuk drop
                if(location.z > 2.0):
                    vz = 0.25
                else:
                    if (location.z > 1.6): #Menurunkan Kecepatan Descent
                         vz = 0.2
                    else:
                         vz = 0
                         if (self.c == True):
                            self.relay2(0) #Capit OFF
                            self.waktu = 0
                            self.c = False
                            self.drop = 1
                            print ("Payload sudah DROP, lanjut misi")
                            sc_logger.text(sc_logger.GENERAL, "Payload sudah DROP, lanjut misi")
                            self.vehicle.commands.next = 5
                            self.controlling_vehicle = False
                            self.vehicle.mode = VehicleMode("AUTO")
                            self.vehicle.flush()
                        
            if active_command == 5: #DROP MP1 atau MP2  
                #self.servo(1)#kamera siap untuk drop
                if(location.z > 2.0):
                    vz = 0.25
                else:
                    if (location.z > 1.6): #Menurunkan Kecepatan Descent
                         vz = 0.2
                    else:
                         vz = 0
                         if (self.c == True):
                            self.relay2(0) #Capit OFF
                            self.waktu = 0
                            self.c = False
                            self.drop = 2
                            print ("Payload sudah DROP, lanjut misi")
                            sc_logger.text(sc_logger.GENERAL, "Payload sudah DROP, lanjut misi")
                            self.controlling_vehicle = False
                            self.vehicle.mode = VehicleMode("AUTO")
                            self.vehicle.flush() 
                        
            if active_command == 6: #DROP LOG
                #self.servo(1)#kamera siap untuk drop
                if(location.z > 2.5):
                    vz = 0.25
                else:
                    if (location.z > 2.0): #Menurunkan Kecepatan Descent
                         vz = 0.15
                    else:
                         vz = 0
                         if (self.c == True):
                            self.relay(1) #buka payload
                            self.waktu = 0
                            self.c = False
                            print ("Payload sudah DROP, lanjut misi")
                            sc_logger.text(sc_logger.GENERAL, "Payload sudah DROP, lanjut misi")
                            self.controlling_vehicle = False
                            self.vehicle.mode = VehicleMode("AUTO")
                            self.vehicle.flush()  	
                            
            if active_command == 7: #PICK MP1
                #Jika ketinggian sudah dibawah 4 meter, maka MAGNET ON
                #self.servo(0)#kamera siap untuk pick
                if (location.z < 4.0):
                    if(location.z > 1.4):
                        vz = 0.15
                    else:
                        vz = 0
                        #active capit
                        if (self.c == True):
                            self.relay2(1) #magnet ON 
                            self.waktu = 0
                            self.c = False
                        print ("Payload sudah diambil, lanjut misi")
                        sc_logger.text(sc_logger.GENERAL, "Payload sudah diambil, lanjut misi")
                        self.controlling_vehicle = False
                        if (self.drop == 2):
                            self.vehicle.commands.next = 8
                            self.lanjut_cmd += 1
                        self.vehicle.mode = VehicleMode("AUTO")
                        self.vehicle.flush()
                                                
            if active_command == 9: #DROP MP2  
                #self.servo(1)#kamera siap untuk drop
                if(location.z > 2.0):
                    vz = 0.25
                else:
                    if (location.z > 1.6): #Menurunkan Kecepatan Descent
                         vz = 0.2
                    else:
                         vz = 0
                         if (self.c == True):
                            self.relay2(0) #magnet OFF
                            self.waktu = 0
                            self.c = False
                            print ("Payload sudah DROP, lanjut misi")
                            sc_logger.text(sc_logger.GENERAL, "Payload sudah DROP, lanjut misi")
                            self.controlling_vehicle = False
                            self.vehicle.mode = VehicleMode("AUTO")
                            self.vehicle.flush()  

            #send velocity commands  toward target heading
            self.send_nav_velocity(vx,vy,vz)

    #move_to_target - fly aircraft to landing pad
    def move_to_target_land(self,target_info,attitude,location):

        x,y = target_info[1]
        
        #pitch_dir,yaw_dir = balloon_finder.pixels_to_direction(x, y, attitude.roll, attitude.pitch, attitude.yaw)
        #yaw_dir = yaw_dir % 360 
        #print "Target Yaw:%f" %(yaw_dir)
        #target_distance = target_info[2]
        #print "Target Distance:%f" %(target_distance*100)
        #shift origin to center of image

        x,y = shift_to_origin((x,y),self.camera_width,self.camera_height)
        hfov = self.camera_hfov
        vfov = self.camera_vfov

        #stabilize image with vehicle attitude
        x -= (self.camera_width / hfov) * math.degrees(attitude.roll)
        y += (self.camera_height / vfov) * math.degrees(attitude.pitch)


        #convert to distance
        X, Y = self.pixel_point_to_position_xy((x,y),location.z)

        #convert to world coordinates
        target_headings = math.atan2(Y,X) #% (2*math.pi)
        target_heading = (attitude.yaw - target_headings) 
        target_distance = math.sqrt(X**2 + Y**2)
     
        sc_logger.text(sc_logger.GENERAL, "Distance to target: {0}".format(round(target_distance,2)))
    
        
        #calculate speed toward target
        speed = target_distance * self.dist_to_vel
        #apply max speed limit
        speed = min(speed,self.vel_speed_max)

        #calculate cartisian speed
        vx = speed * math.sin(target_heading)*  -1.0
        vy = speed * math.cos(target_heading) #* -1.0

        print "Found Target go to vx:%f vy:%f Alt:%f target_distance:%f" % (vx,vy,location.z,target_distance)

        #only descend when on top of target
        if(target_distance < self.descent_radius):
            vz = 0
        else:
            vz = 0

        #send velocity commands toward target heading
        self.send_nav_velocity(vx,vy,vz)

    #move_to_target - fly aircraft to landing pad
    def move_to_target(self,target_info,attitude,location):

        x,y = target_info[1]
        
        #pitch_dir,yaw_dir = balloon_finder.pixels_to_direction(x, y, attitude.roll, attitude.pitch, attitude.yaw)
        #yaw_dir = yaw_dir % 360 
        #print "Target Yaw:%f" %(yaw_dir)
        #target_distance = target_info[2]
        #print "Target Distance:%f" %(target_distance*100)
        #shift origin to center of image

        x,y = shift_to_origin((x,y),self.camera_width,self.camera_height)
        hfov = self.camera_hfov
        vfov = self.camera_vfov

        #stabilize image with vehicle attitude
        x -= (self.camera_width / hfov) * math.degrees(attitude.roll)
        y += (self.camera_height / vfov) * math.degrees(attitude.pitch)


        #convert to distance
        X, Y = self.pixel_point_to_position_xy((x,y),location.z)

        #convert to world coordinates
        target_headings = math.atan2(Y,X) #% (2*math.pi)
        target_heading = (attitude.yaw - target_headings) 
        target_distance = math.sqrt(X**2 + Y**2)
     
        sc_logger.text(sc_logger.GENERAL, "Distance to target: {0}".format(round(target_distance,2)))
    
        
        #calculate speed toward target
        speed = target_distance * self.dist_to_vel
        #apply max speed limit
        speed = min(speed,self.vel_speed_max)

        #calculate cartisian speed
        vx = speed * math.sin(target_heading) * -1.0
        vy = speed * math.cos(target_heading) 

        print "Found Target go to vx:%f vy:%f Alt:%f target_distance:%f" % (vx,vy,location.z,target_distance)

        #only descend when on top of target
        if(target_distance > self.descent_radius):
            vz = 0
        else:
            vz = self.descent_rate

        #send velocity commands toward target heading
        self.send_nav_velocity(vx,vy,vz)

	#move_to_target - fly aircraft to landing pad
    def move_to_target_cam_forward(self,target_info,attitude,location):
        x,y = target_info[1]
        pitch_dir,yaw_dir = balloon_finder.pixels_to_direction(x, y, attitude.roll, attitude.pitch, attitude.yaw)
        yaw_dir = yaw_dir % 360 
        print "Target Yaw:%f" %(yaw_dir)
        target_distance = target_info[2]*10
        print "Target Distance:%f" %(target_distance)
        #shift origin to center of image
      
        print "Found Target at yaw:%f heading:%f Alt:%f Dist:%f" % (attitude.yaw,yaw_dir,location.z,target_distance)

        sc_logger.text(sc_logger.GENERAL, "Distance to target: {0}".format(round(target_distance,2)))

        print "mode detection"
        #self.condition_yaw(math.fabs(math.degrees(target_heading)))
        if(target_distance > 4):
            vx = 0
            vy = 0 
            vz = 0
        else:
            vx = 0
            vy = 0
            vz = 0
        #send velocity commands toward target heading
        self.send_nav_velocity(vx,vy,vz)
        self.condition_yaw(yaw_dir)

    #inside_landing_area - determine is we are in a landing zone 0 = False, 1 = True, -1 = below the zone
    def inside_landing_area(self):

        vehPos = PositionVector.get_from_location(self.vehicle.location.global_relative_frame)
        landPos = PositionVector.get_from_location(PositionVector.get_home_location())
        '''
        vehPos = PositionVector.get_from_location(Location(0,0,10))
        landPos = PositionVector.get_from_location(Location(0,0,0))
        '''
        if(PositionVector.get_distance_xy(vehPos,landPos) < self.landing_area_radius):
            #below area
            if(vehPos.z < self.landing_area_min_alt):
                print "VehPos.Z: %f " % (vehPos.z)
                return 1
            #in area
            else:
                return 1
        #outside area
        else:
            return 0

    #pixel_point_to_position_xy - convert position in pixels to position in meters
    #pixel_position - distance in pixel from CENTER of image
    #distance- distance from the camera to the object in  meters 
    def pixel_point_to_position_xy(self,pixel_position,distance):
        thetaX = pixel_position[0] * self.camera_hfov / self.camera_width
        thetaY = pixel_position[1] * self.camera_vfov / self.camera_height
        x = distance * math.tan(math.radians(thetaX))
        y = distance * math.tan(math.radians(thetaY))

        return (x,y)

    def run(self):
        while True:

            now = time.time()

            # only process images once home has been initialised
            if self.check_home():

                # start video if required
                self.check_video_out()
    
                # check if we are controlling the vehicle
                self.check_status()

                if self.mission_on_guide_mode == 4:
                    self.analisis_image()

                # output stats
                self.output_frame_rate_stats()

            # Don't suck up too much CPU, only process a new image occasionally
            time.sleep(0.05)

        if not self.use_simulator:
            balloon_video.stop_background_capture()

    # complete - balloon strategy has somehow completed so return control to the autopilot
    def complete(self):
        # debug
        if self.debug:
            print "Complete!"

        # stop the vehicle and give up control
        if self.controlling_vehicle:
            self.guided_target_vel = (0,0,0)
            self.send_nav_velocity(self.guided_target_vel[0], self.guided_target_vel[1], self.guided_target_vel[2])
            self.guided_last_update = time.time()

        # if in GUIDED mode switch back to LOITER
        if self.vehicle.mode.name == "GUIDED":
            self.vehicle.mode = VehicleMode("LOITER")
            self.vehicle.flush()

        # if in AUTO move to next command
        if self.vehicle.mode.name == "AUTO":
            self.advance_current_cmd();

        # flag we are not in control of the vehicle
        self.controlling_vehicle = False
            
        return

strat = BalloonStrategy()
strat.run()

