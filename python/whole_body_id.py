from xbot2_interface import pyxbot2_interface as xbi

import numpy as np
from ttictoc import tic,toc

import rospy
import tf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, WrenchStamped

from pyopensot.tasks.acceleration import CoM, Cartesian, Postural, DynamicFeasibility
from pyopensot.constraints.acceleration import JointLimits, VelocityLimits
from pyopensot.constraints.force import FrictionCone
import pyopensot as pysot


rospy.init_node("g1_wbid", disable_signals=True)
dt = 1.0 / 500

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

dq = np.zeros(model.nv)
model.setJointPosition(q)
model.setJointVelocity(dq)
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
variables = pysot.OptvarHelper(variables_vec)

# Set CoM tracking task
com = CoM(model, variables.getVariable("qddot"))
com.setLambda(1.)
# FK at initial config
com_ref, vel_ref, acc_ref = com.getReference()
com0 = com_ref.copy()

# Set the whole Cartesian task but later only orientation will be used
base = Cartesian("base", model, "world", "pelvis", variables.getVariable("qddot"))
base.setLambda(1.)

# Set the contact task
contact_tasks = list()
cartesian_contact_task_frames = ["left_foot_point_contact", "right_foot_point_contact"]
for cartesian_contact_task_frame in cartesian_contact_task_frames:
  contact_tasks.append(Cartesian(cartesian_contact_task_frame, model, cartesian_contact_task_frame, "world", variables.getVariable("qddot")))

posture = Postural(model, variables.getVariable("qddot"))

# For the base task taking only the orientation part
stack = (0.1*com + 0.1*(base%[3, 4, 5]))
force_variables = list()

for i in range(len(cartesian_contact_task_frames)):
  stack = stack + 10.*(contact_tasks[i])

for i in range(len(contact_frames)):
  force_variables.append(variables.getVariable(contact_frames[i]))


# Creates the stack.
# Notice:  we do not need to keep track of the DynamicFeasibility constraint so it is created when added into the stack.
# The same can be done with other constraints such as Joint Limits and Velocity Limits
stack = (pysot.AutoStack(stack)/posture) << DynamicFeasibility("floating_base_dynamics", model, variables.getVariable("qddot"), force_variables, contact_frames)
stack = stack << JointLimits(model, variables.getVariable("qddot"), qmax, qmin, 10.*dqmax, dt)
stack = stack << VelocityLimits(model, variables.getVariable("qddot"), dqmax, dt)
for i in range(len(contact_frames)):
	T = model.getPose(contact_frames[i])
	mu = (T.linear, 0.8) # rotation is world to contact
	stack = stack << FrictionCone(contact_frames[i], variables.getVariable(contact_frames[i]), model, mu)
   
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

force_msg = list()
fpubs = list()
for contact_frame in contact_frames:
	force_msg.append(WrenchStamped())
	force_msg[-1].header.frame_id = contact_frame
	force_msg[-1].wrench.torque.x = force_msg[-1].wrench.torque.y = force_msg[-1].wrench.torque.z = 0.
	fpubs.append(rospy.Publisher(contact_frame, WrenchStamped, queue_size=1))

t = 0.
alpha = 0.4
while not rospy.is_shutdown():
	tic()
	# Update actual position in the model
	model.setJointPosition(q)
	model.setJointVelocity(dq)
	model.update()

	# Compute new reference for CoM task
	com_ref[2] = com0[2] + alpha * np.sin(3.1415 * t)
	com_ref[1] = com0[1] + alpha * np.cos(3.1415 * t)
	t = t + dt
	com.setReference(com_ref)

	# Update Stack
	stack.update()

	# Solve
	x = solver.solve()
	ddq = variables.getVariable("qddot").getValue(x) # from variables vector we retrieve the joint accelerations
	q = model.sum(q, dq*dt + 0.5 * ddq * dt * dt) # we use the model sum to account for the floating-base
	dq += ddq*dt

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
  
	# Time unit is seconds
	# print("Time to compute: ", toc())

	rate.sleep()