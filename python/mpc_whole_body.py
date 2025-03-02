import sys
import os
# To be able to import srbd_mpc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from xbot2_interface import pyxbot2_interface as xbi
from srbd_mpc import mpc
from srbd_mpc import gait_planner

import numpy as np

import rospy
import tf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, WrenchStamped

from pyopensot.tasks.acceleration import CoM, Cartesian, DynamicFeasibility, Postural
from pyopensot.constraints.acceleration import JointLimits, VelocityLimits
from pyopensot.constraints.force import FrictionCone
import pyopensot as pysot

import matplotlib.pyplot as plt

def MinimizeVariable(name, opt_var): 
	'''Task to regularize a variable using a generic task'''
	A = opt_var.getM()

	# The minus because y = Mx + q is mapped on ||Ax - b|| 
	b = -opt_var.getq()
	task = pysot.GenericTask(name, A, b, opt_var)


	# Setting the regularization weight.
	task.setWeight(1.)
	task.update()

	# print(f"MinVar A:\n {task.getA()}")
	# print(f"MinVar b:\n {task.getb()}")
	# print(f"MinVar W:\n {task.getWeight()}")

	return task

def Wrench(name, distal_link, base_link, wrench):
	'''Task to minimize f-fd using a generic task'''

	# print(wrench)
	# print(wrench.getM().shape)
	# print(wrench.getq())

	A = np.eye(3)
	b =	- wrench.getq()
	
	return pysot.GenericTask(name, A, b, wrench) 

def setDesiredForce(Wrench_task, wrench_desired, wrench):
	# b = -(wrench - wrench_desired).getq()

	b = wrench_desired - wrench.getq()
	
	print(f"b: {b}")
	
	# print(f'wrench_desired_dimensions: {wrench_desired.shape}')
	# print(f'wrench: {wrench}')
	# print(wrench.getM().shape)
	# print(f'wrench_dimensions: {wrench.getq().shape}')
	# print(f'b_dimensions: {b.shape}')
	Wrench_task.setb(b)

def plot_all_trajectories(com_hist, orient_hist, vel_hist, orient_deriv_hist, 
                          com_ref_hist, orient_ref_hist, vel_ref_hist, orient_deriv_ref_hist, dt):
    n_points = len(com_hist)
    time_array = np.arange(n_points) * dt

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Center of Mass Trajectory
    ax = axs[0, 0]
    ax.plot(time_array, [c[0] for c in com_hist], label="X")
    ax.plot(time_array, [c[1] for c in com_hist], label="Y")
    ax.plot(time_array, [c[2] for c in com_hist], label="Z")
    
    # Add reference trajectories as dotted lines
    ax.plot(time_array, [c[0] for c in com_ref_hist], 'b--', label="X ref")
    ax.plot(time_array, [c[1] for c in com_ref_hist], '--', color='orange', label="Y ref")
    ax.plot(time_array, [c[2] for c in com_ref_hist], 'g--', label="Z ref")
    
    ax.set_title("Center of Mass Trajectory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)")
    ax.grid(True)
    ax.legend()

    # Orientation Trajectory
    ax = axs[0, 1]
    ax.plot(time_array, [o[0] for o in orient_hist], label="Roll")
    ax.plot(time_array, [o[1] for o in orient_hist], label="Pitch")
    ax.plot(time_array, [o[2] for o in orient_hist], label="Yaw")
    
    # Add reference trajectories as dotted lines
    ax.plot(time_array, [o[0] for o in orient_ref_hist], 'b--', label="Roll ref")
    ax.plot(time_array, [o[1] for o in orient_ref_hist], '--', color='orange', label="Pitch ref")
    ax.plot(time_array, [o[2] for o in orient_ref_hist], 'g--', label="Yaw ref")
    
    ax.set_title("SRBD Orientation")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.grid(True)
    ax.legend()

    # COM Velocity Trajectory
    ax = axs[1, 0]
    ax.plot(time_array, [v[0] for v in vel_hist], label="Vx")
    ax.plot(time_array, [v[1] for v in vel_hist], label="Vy")
    ax.plot(time_array, [v[2] for v in vel_hist], label="Vz")
    
    # Add reference trajectories as dotted lines
    ax.plot(time_array, [v[0] for v in vel_ref_hist], 'b--', label="Vx ref")
    ax.plot(time_array, [v[1] for v in vel_ref_hist], '--', color='orange', label="Vy ref")
    ax.plot(time_array, [v[2] for v in vel_ref_hist], 'g--', label="Vz ref")
    
    ax.set_title("COM Velocity Trajectory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.grid(True)
    ax.legend()

    # Orientation Derivative Trajectory
    ax = axs[1, 1]
    ax.plot(time_array, [od[0] for od in orient_deriv_hist], label="Roll Rate")
    ax.plot(time_array, [od[1] for od in orient_deriv_hist], label="Pitch Rate")
    ax.plot(time_array, [od[2] for od in orient_deriv_hist], label="Yaw Rate")
    
    # Add reference trajectories as dotted lines
    ax.plot(time_array, [od[0] for od in orient_deriv_ref_hist], 'b--', label="Roll Rate ref")
    ax.plot(time_array, [od[1] for od in orient_deriv_ref_hist], '--', color='orange', label="Pitch Rate ref")
    ax.plot(time_array, [od[2] for od in orient_deriv_ref_hist], 'g--', label="Yaw Rate ref")
    
    ax.set_title("Orientation Derivative Trajectory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate (rad/s)")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()
    
def plot_ground_reaction_forces(grf_hist, dt):
    time_array = np.arange(len(grf_hist)) * dt
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Left Heel (Back) GRF: columns 0-2
    axs[0, 0].plot(time_array, [f[0] for f in grf_hist], label='Fx Heel (Left)')
    axs[0, 0].plot(time_array, [f[1] for f in grf_hist], label='Fy Heel (Left)')
    axs[0, 0].plot(time_array, [f[2] for f in grf_hist], label='Fz Heel (Left)')
    axs[0, 0].set_title('Left Heel Ground Reaction Forces')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Force (N)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Left Toe (Front) GRF: columns 3-5
    axs[0, 1].plot(time_array, [f[3] for f in grf_hist], label='Fx Toe (Left)')
    axs[0, 1].plot(time_array, [f[4] for f in grf_hist], label='Fy Toe (Left)')
    axs[0, 1].plot(time_array, [f[5] for f in grf_hist], label='Fz Toe (Left)')
    axs[0, 1].set_title('Left Toe Ground Reaction Forces')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Force (N)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # Right Heel (Back) GRF: columns 6-8
    axs[1, 0].plot(time_array, [f[6] for f in grf_hist], label='Fx Heel (Right)')
    axs[1, 0].plot(time_array, [f[7] for f in grf_hist], label='Fy Heel (Right)')
    axs[1, 0].plot(time_array, [f[8] for f in grf_hist], label='Fz Heel (Right)')
    axs[1, 0].set_title('Right Heel Ground Reaction Forces')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Force (N)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # Right Toe (Front) GRF: columns 9-11
    axs[1, 1].plot(time_array, [f[9] for f in grf_hist], label='Fx Toe (Right)')
    axs[1, 1].plot(time_array, [f[10] for f in grf_hist], label='Fy Toe (Right)')
    axs[1, 1].plot(time_array, [f[11] for f in grf_hist], label='Fz Toe (Right)')
    axs[1, 1].set_title('Right Toe Ground Reaction Forces')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Force (N)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

rospy.init_node("g1_wbid", disable_signals=True)

# dt = 1.0 / 500

# For MPC to work
dt = 0.04

br = tf.TransformBroadcaster()

# Get model from urdf
urdf = rospy.get_param("/robot_description")
model = xbi.ModelInterface2(urdf)
qmin, qmax = model.getJointLimits()
dqmax = model.getVelocityLimits()

# Initial Joint Configuration
q = [
    # floating base
    0.0,
    0.0,
    0.0,  # reference base linear
    0.0,
    0.0,
    0.0,
    1.0,  # refernce base quaternion
    ## left leg
    -0.6,  # left_hip_pitch_joint
    0.0,  # left_hip_roll_joint
    0.0,  # left_hip_yaw_joint
    1.2,  # left_knee_joint
    -0.6,  # left_ankle_pitch_joint
    0.0,  # left_ankle_roll_joint
    ## right leg
    -0.6,  # right_hip_pitch_joint
    0.0,  # right_hip_roll_joint
    0.0,  # right_hip_yaw_joint
    1.2,  # right_knee_joint
    -0.6,  # right_ankle_pitch_joint
    0.0,  # right_ankle_roll_joint
    ## waist
    0.0,  # waist_yaw_joint
    ## left shoulder
    0.0,  # left_shoulder_pitch_joint
    0.0,  # left_shoulder_roll_joint
    0.0,  # left_shoulder_yaw_joint
    0.0,  # left_elbow_joint
    0.0,  # left_wrist_roll_joint
    ## right shoulder
    0.0,  #'right_shoulder_pitch_joint'
    0.0,  # right_shoulder_roll_joint
    0.0,  # right_shoulder_yaw_joint
    0.0,  # right_elbow_joint
    0.0,  # right_wrist_roll_joint
]

# Set initial positions and velocities in the model
dq = np.zeros(model.nv)
model.setJointPosition(q)
model.setJointVelocity(dq)

# Update model to do Forward Kinematics and other updates
model.update()

# Instantiate Variables: qddot and contact forces (3 per contact)
variables_vec = dict()
variables_vec["qddot"] = model.nv

line_foot_contact_frames = ["left_foot_line_contact_lower", "left_foot_line_contact_upper",
                            "right_foot_line_contact_lower", "right_foot_line_contact_upper"]

# Hands may be added in the future
contact_frames = line_foot_contact_frames

for contact_frame in contact_frames:
  variables_vec[contact_frame] = 3

# Defining optimization variables
variables = pysot.OptvarHelper(variables_vec)

# Just to get the com ref i will not use the task
com = CoM(model, variables.getVariable("qddot"))
com.setLambda(1.)
com_ref, vel_ref, acc_ref = com.getReference()
com0 = com_ref.copy()

# # Set the whole Cartesian task but later only orientation will be used
base = Cartesian("base", model, "world", "pelvis", variables.getVariable("qddot"))
base.setLambda(1.)


# Set the contact task
contact_tasks = list()
cartesian_contact_task_frames = ["left_foot_point_contact", "right_foot_point_contact"]
for cartesian_contact_task_frame in cartesian_contact_task_frames:
	contact_tasks.append(Cartesian(cartesian_contact_task_frame, model, cartesian_contact_task_frame, "world", variables.getVariable("qddot")))
	if "stack" not in globals():
		stack = 10.*(contact_tasks[-1])
	stack = stack + 10.*(contact_tasks[-1])

# stack = stack + 0.1*com + 0.1*(base%[3, 4, 5])
# stack = stack + 0.1*com
# stack = stack + 0.1*(base%[3, 4, 5])

# # Postural task
posture = Postural(model, variables.getVariable("qddot"))
posture.setLambda(1.)


force_variables = list()

# Task for factual - fdesired
wrench_tasks = list()
for contact_frame in contact_frames:
	wrench_tasks.append(Wrench(contact_frame, contact_frame, "pelvis", variables.getVariable(contact_frame)))
	stack = stack + 0.01*(wrench_tasks[-1])

for i in range(len(contact_frames)):
	force_variables.append(variables.getVariable(contact_frames[i]))

# Adds the floating base dynamics constraint
stack = (pysot.AutoStack(stack)/posture) << DynamicFeasibility("floating_base_dynamics", model, variables.getVariable("qddot"), force_variables, contact_frames)
# Adds joint limits and velocity limits constraints
stack = stack << JointLimits(model, variables.getVariable("qddot"), qmax, qmin, 10.*dqmax, dt)
stack = stack << VelocityLimits(model, variables.getVariable("qddot"), dqmax, dt)


# Adds the friction cones constraints
for i in range(len(contact_frames)):
	T = model.getPose(contact_frames[i])
	# Note: T.linear is the rotation from world to contact not the translation
	mu = (T.linear, 0.3) # rotation is world to contact
	stack = stack << FrictionCone(contact_frames[i], variables.getVariable(contact_frames[i]), model, mu)

# Regularization task
reg_qddot = MinimizeVariable("req_qddot", variables.getVariable("qddot"))

# The regularization task should be added this ways
# otherwise if added with + in stack it will not be consider in all priority levels
stack.setRegularisationTask((1e-3)*reg_qddot)	 

# for contact_frame in contact_frames:
	# stack.setRegularisationTask((1e-5)*MinimizeVariable(contact_frame, variables.getVariable(contact_frame)))

# Creates the solver	
solver = pysot.iHQP(stack)

# ID loop: we publish also joint position, floating-base pose and contact forces
rate = rospy.Rate(1./dt)
pub = rospy.Publisher('joint_states', JointState, queue_size=1)
msg = JointState()
msg.name = model.getJointNames()[1::]

w_T_b = TransformStamped()
w_T_b.header.frame_id = "world"
w_T_b.child_frame_id = "pelvis"

# Create publishers for contact forces
force_msg = list()
fpubs = list()
for contact_frame in contact_frames:
	force_msg.append(WrenchStamped())
	force_msg[-1].header.frame_id = contact_frame
	force_msg[-1].wrench.torque.x = force_msg[-1].wrench.torque.y = force_msg[-1].wrench.torque.z = 0.
	fpubs.append(rospy.Publisher(contact_frame, WrenchStamped, queue_size=1))

# MPC init
SRBD_mpc = mpc.MPC(dt = dt)
SRBD_mpc.init_matrices()

# Setup reference horizon.
SRBD_mpc.x_ref_hor[:, -1] = SRBD_mpc.g

com_hist = []
orient_hist = []
vel_hist = []
orient_deriv_hist = []
grf_hist = []

com_ref_hist = []
orient_ref_hist = []
vel_ref_hist = []
orient_deriv_ref_hist = []

t = 0.
alpha = 0.4
while not rospy.is_shutdown():
	# Update actual position in the model
	model.setJointPosition(q)
	model.setJointVelocity(dq)
	model.update()

	# Compute new reference for CoM task
	# com_ref[1] = com0[1] + alpha * np.cos(3.1415 * t)
	# com_ref[2] = com0[2] + alpha * np.sin(3.1415 * t)
	# print(f"com_ref: {com_ref}")
	# com.setReference(com_ref)

	# Feed current CoM to MPC
	# SRBD_mpc.x_ref_hor[0, 3:6] = com0.copy()

	if t == 0.0:
		# Feed current Base orientation to MPC
		q_euler0 = np.array(tf.transformations.euler_from_quaternion(q[3:7]))
		SRBD_mpc.x_ref_hor[0, 0:3] = q_euler0.copy()
		SRBD_mpc.x_ref_hor[0, 3:6] = com0.copy()
		SRBD_mpc.x0 = SRBD_mpc.x_ref_hor[0].copy()
	else:
		SRBD_mpc.x0 = x_current.copy()
	
	# Feed horizon	CoM to MPC
	for i in range(1, SRBD_mpc.HORIZON_LENGTH-1):
		# com_ref[1] = com0[1] + alpha * np.cos(3.1415 * (t + i*dt))
		# com_ref[2] = com0[2] + alpha * np.sin(3.1415 * (t + i*dt))	

		com_ref[0] = 5.26790425e-02
		com_ref[1] = 7.44339342e-05
		com_ref[2] = -8.20167454e-02
		SRBD_mpc.x_ref_hor[i, 3] = com_ref[0].copy() # x position
		SRBD_mpc.x_ref_hor[i, 4] = com_ref[1].copy() # y position
		SRBD_mpc.x_ref_hor[i, 5] = com_ref[2].copy() # z position     

	
	# Feed current Base orientation to MPC
	q_euler0 = np.array(tf.transformations.euler_from_quaternion(q[3:7]))

	# Transform Base orientation from quaternion to euler angles
	SRBD_mpc.x_ref_hor[0, 0:3] = q_euler0.copy()
	
	# Feed horizon Base to MPC
	# euler_angles = np.array(tf.transformations.euler_from_quaternion(q[3:7]))
	euler_angles = np.array([0., 0., 0.])
	SRBD_mpc.x_ref_hor[1:, 0:3] = np.tile(euler_angles, (SRBD_mpc.HORIZON_LENGTH-1, 1))
	
	# Current foot heel and toe positions
	left_heel = model.getPose("left_foot_line_contact_lower").translation
	left_toe = model.getPose("left_foot_line_contact_upper").translation
	
	right_heel = model.getPose("right_foot_line_contact_lower").translation
	right_toe = model.getPose("right_foot_line_contact_upper").translation

	c_horizon = []
	contact_horizon = []
	for i in range(SRBD_mpc.HORIZON_LENGTH):
		c_horizon.append(np.concatenate((left_heel, left_toe, right_heel ,right_toe)))
	
	# Both feet in contact for all the horizon
	for i in range(SRBD_mpc.HORIZON_LENGTH):
		contact_horizon.append(np.array([1, 1, 1, 1]))

	t = t + dt

	p_com_horizon = SRBD_mpc.x_ref_hor[:, 3:6].copy()

	# Perform MPC calculations.
	SRBD_mpc.extract_psi()
	SRBD_mpc.rotation_matrix_T()
	SRBD_mpc.set_Q()
	SRBD_mpc.set_R()
	SRBD_mpc.calculate_A_continuous()
	SRBD_mpc.calculate_A_discrete()
	SRBD_mpc.calculate_B_continuous(c_horizon, p_com_horizon)
	SRBD_mpc.calculate_B_discrete()
	SRBD_mpc.calculate_Aqp()
	SRBD_mpc.calculate_Bqp()
	SRBD_mpc.calculate_Ac()
	SRBD_mpc.calculate_bounds(contact_horizon)
	SRBD_mpc.calculate_hessian()
	SRBD_mpc.calculate_gradient()
	SRBD_mpc.solve_qp()

	# Taking the first control input
	u_opt0 = SRBD_mpc.u_opt[0, :]

	# Rollout the control input
	SRBD_mpc.compute_rollout(SRBD_mpc.x0)

	
	# I want to divide the weight of the robot in the contact points
	# gravity = 9.80665
	# wrench_desired = np.array([0., 0., model.getMass()*gravity / len(contact_frames)])
	
	# wrench_desired_heel_left = np.array([0., 0., 88.])
	# wrench_desired_toe_left = np.array([0., 0., 82.])
	# wrench_desired_heel_right = np.array([0., 0., 88.])
	# wrench_desired_toe_right = np.array([0., 0., 82.])

	# wrench_concatenated = np.vstack((wrench_desired_heel_left, wrench_desired_toe_left, wrench_desired_heel_right, wrench_desired_toe_right))

	# u_opt0 has form of [fx_left_heel, fy_left_heel, fz_left_heel, fx_left_toe, fy_left_toe, fz_left_toe, fx_right_heel, fy_right_heel, fz_right_heel, fx_right_toe, fy_right_toe, fz_right_toe]
	
	for i in range(len(contact_frames)):
			setDesiredForce(wrench_tasks[i], u_opt0[i*3:i*3+3], variables.getVariable(contact_frames[i]))
			# setDesiredForce(wrench_tasks[i], wrench_desired, variables.getVariable(contact_frames[i]))
	
	# Update Stack
	stack.update()
	
	# Solve WBID QP
	x = solver.solve()
	
	ddq = variables.getVariable("qddot").getValue(x) # from variables vector we retrieve the joint accelerations
	q = model.sum(q, dq*dt + 0.5 * ddq * dt * dt) # we use the model sum to account for the floating-base
	dq += ddq*dt

	# Print me heel and toe forces from the solver
	for i in range(len(contact_frames)):
		print("QP solver =========================================")
		print(f"contact_frame: {contact_frames[i]}, force: {variables.getVariable(contact_frames[i]).getValue(x)}")

	
	# Update current state
	com = CoM(model, variables.getVariable("qddot"))
	com_ref, _, _ = com.getReference()
	com0 = com_ref.copy()
	
	q_euler0 = tf.transformations.euler_from_quaternion(q[3:7])
	# print(f"q_euler0: {q_euler0}")
	
	# Getting the orientaion speed of the base since CoM does not have orientation
	dq_euler0 = SRBD_mpc.rotation_z_T @ dq[3:6]
	
	vel0_ref = dq[0:3]

	# Print me the shapes of rotation and dq
	# print(f"rotation_z_T shape: {SRBD_mpc.rotation_z_T.shape}")
	# print(f"dq shape: {dq[3:6].shape}")


	# print(f"dq_euler0: {dq_euler0}")

	x_current = np.zeros(SRBD_mpc.NUM_STATES)
	x_current[0:3] = q_euler0
	x_current[3:6] = com0
	x_current[6:9] = dq_euler0
	x_current[9:12] = vel0_ref
	x_current[12] = SRBD_mpc.g

	# Add the gravity as the last term in x_current


	if "pass_count" not in globals():
		pass_count = 0
	
	pass_count += 1
	print(f"pass_count: {pass_count}")
	if pass_count >= 10:
		# Convert lists to numpy arrays for easier plotting
		plot_all_trajectories(com_hist, orient_hist, vel_hist, orient_deriv_hist, com_ref_hist, orient_ref_hist, vel_ref_hist, orient_deriv_ref_hist, dt)
		plot_ground_reaction_forces(grf_hist, dt)
		exit()	

	# Record current state in history
	com_hist.append(com0.copy())
	orient_hist.append(q_euler0)
	vel_hist.append(vel0_ref.copy())
	orient_deriv_hist.append(dq_euler0.copy())
	grf_hist.append(u_opt0.copy())

	# Record reference state in history
	com_ref_hist.append(SRBD_mpc.x_ref_hor[1, 3:6].copy())
	orient_ref_hist.append(SRBD_mpc.x_ref_hor[1, 0:3].copy())
	vel_ref_hist.append(SRBD_mpc.x_ref_hor[1, 9:12].copy())
	orient_deriv_ref_hist.append(SRBD_mpc.x_ref_hor[1, 6:9].copy())

	# Publish joint states
	msg.position = q[7::]
	msg.header.stamp = rospy.get_rostime()

	w_T_b.header.stamp = msg.header.stamp
	w_T_b.transform.translation.x = q[0]
	w_T_b.transform.translation.y = q[1]
	w_T_b.transform.translation.z = q[2]
	w_T_b.transform.rotation.x = q[3]
	w_T_b.transform.rotation.y = q[4]
	w_T_b.transform.rotation.z = q[5]
	w_T_b.transform.rotation.w = q[6]

	br.sendTransformMessage(w_T_b)

	for i in range(len(contact_frames)):
		T = model.getPose(contact_frames[i])
		force_msg[i].header.stamp = msg.header.stamp
		f_local = T.linear.transpose() @ variables.getVariable(contact_frames[i]).getValue(x) # here we compute the value of the contact forces in local frame from world frame
		force_msg[i].wrench.force.x = f_local[0]
		force_msg[i].wrench.force.y = f_local[1]
		force_msg[i].wrench.force.z = f_local[2]
		fpubs[i].publish(force_msg[i])


	pub.publish(msg)

	rate.sleep()