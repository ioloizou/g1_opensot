import os

import rospkg
from xbot2_interface import pyxbot2_interface as xbi
from pyopensot.tasks.acceleration import (
    Cartesian,
    CoM,
    DynamicFeasibility,
    Postural,
    AngularMomentum,
)
from pyopensot.constraints.acceleration import JointLimits, VelocityLimits
from pyopensot.constraints.force import FrictionCone
import pyopensot as pysot
import numpy as np

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, WrenchStamped
import tf
import subprocess
from ttictoc import tic, toc

rospy.init_node("g1_example", disable_signals=True)
dt = 1.0 / 500.0

br = tf.TransformBroadcaster()

# Get robot description parameter and initialize model interface (with Pinocchio)
urdf = rospy.get_param("/robot_description")
model = xbi.ModelInterface2(urdf)
qmin, qmax = model.getJointLimits()
dqmax = model.getVelocityLimits()

# print(model.getJointNames())

# Initialise the model
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
    ## right hand
    0.0,  #'right_shoulder_pitch_joint'
    0.0,  # right_shoulder_roll_joint
    0.0,  # right_shoulder_yaw_joint
    0.0,  # right_elbow_joint
    0.0,  # right_wrist_roll_joint
    # 'left_shoulder_pitch_joint'
    # 'left_shoulder_roll_joint'
    # 'left_shoulder_yaw_joint'
    # 'left_elbow_joint'
    # 'left_wrist_roll_joint'
    # 'left_wrist_pitch_joint'
    # 'left_wrist_yaw_joint'
    ## right shoulder
    # 'right_shoulder_pitch_joint'
    # 'right_shoulder_roll_joint'
    # 'right_shoulder_yaw_joint'
    # 'right_elbow_joint'
    # 'right_wrist_roll_joint'
    # 'right_wrist_pitch_joint'
    # 'right_wrist_yaw_joint'
]

print(model.nq)
print(model.getJointNames())
exit()

dq = np.zeros(model.nv)
model.setJointPosition(q)
model.setJointVelocity(dq)
model.update()

foot_contact_frames = [
    "left_foot_upper_right",
    "left_foot_lower_right",
    "left_foot_upper_left",
    "left_foot_lower_left",
    "right_foot_upper_right",
    "right_foot_lower_right",
    "right_foot_upper_left",
    "right_foot_lower_left",
]

# Line contact
# foot_contact_frames = [
#     "left_foot_line_contact_upper",
#     "left_foot_line_contact_lower",
#     "right_foot_line_contact_upper",
#     "right_foot_line_contact_lower",
# ]

# Point contact
# foot_contact_frames = ["left_foot_point_contact", "right_foot_point_contact"]

hand_contact_frames = ["left_hand_point_contact", "right_hand_point_contact"]

contact_frames = foot_contact_frames
# contact_frames = foot_contact_frames + hand_contact_frames

variables_vec = dict()
variables_vec["qddot"] = model.nv

variables = pysot.OptvarHelper(variables_vec)

# Create task and constraints
# CoM Task
com = CoM(model, variables.getVariable("qddot"))
com.setLambda(1.0)
com_ref, vel_ref, acc_ref = com.getReference()
com0 = com_ref.copy()

contact_tasks = list()
for contact_frame in contact_frames:
    contact_tasks.append(
        Cartesian(
            contact_frame,
            model,
            contact_frame,
            "world",
            variables.getVariable("qddot"),
        )
    )

hand_contact_tasks = list()
for hand_contact_frame in hand_contact_frames:
    hand_contact_tasks.append(
        Cartesian(
            hand_contact_frame,
            model,
            hand_contact_frame,
            "pelvis",
            variables.getVariable("qddot"),
        )
    )

# Cartesian Task
base = Cartesian("base", model, "world", "pelvis", variables.getVariable("qddot"))
base.setLambda(0.8)

# Postural Task
posture = Postural(model, variables.getVariable("qddot"))

# Angular Momentum Task
angular = AngularMomentum(model, variables.getVariable("qddot"))

# Create Stack of Tasks
stack = 0.1 * com + 0.1 * base % [3, 4, 5] + 0.00001 * posture + 0.1 * angular
for i in range(len(contact_frames)):
    stack = stack + 10.0 * (contact_tasks[i])
# for i in range(len(hand_contact_frames)):
#     stack = stack + 10.0 * (hand_contact_tasks[i] % [0, 1, 2])

stack = pysot.AutoStack(stack)
stack = stack << JointLimits(
    model, variables.getVariable("qddot"), qmax, qmin, 10.0 * dqmax, dt
)
stack = stack << VelocityLimits(model, variables.getVariable("qddot"), dqmax, dt)

# Creates the solver
solver = pysot.iHQP(stack)

rate = rospy.Rate(1.0 / dt)
pub = rospy.Publisher("joint_states", JointState, queue_size=1)
msg = JointState()
msg.name = model.getJointNames()[1::]

w_T_b = TransformStamped()
w_T_b.header.frame_id = "world"
w_T_b.child_frame_id = "pelvis"

t = 0.0
alpha = 0.4
freq = 1.5
while not rospy.is_shutdown():
    # Update actual position in the model
    model.setJointPosition(q)
    model.setJointVelocity(dq)
    model.update()

    # Compute new reference for CoM task
    com_ref[0] = com0[0] + alpha * np.sin(2 * 3.1415 * freq * t)
    com_ref[1] = com0[1] + alpha * np.cos(2 * 3.1415 * freq * t)
    com_ref[2] = com0[2] + alpha * np.cos(2 * 3.1415 * freq * t)
    t = t + dt
    com.setReference(com_ref)

    # Update Stack
    stack.update()

    # Solve
    x = solver.solve()
    ddq = variables.getVariable("qddot").getValue(
        x
    )  # from variables vector we retrieve the joint accelerations
    q = model.sum(
        q, dq * dt + 0.5 * ddq * dt * dt
    )  # we use the model sum to account for the floating-base
    dq += ddq * dt

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

    #
    #     for i in range(len(contact_frames)):
    #         T = model.getPose(contact_frames[i])
    #         force_msg[i].header.stamp = msg.header.stamp
    #         f_local = T.linear.transpose() @ variables.getVariable(
    #             contact_frames[i]
    #         ).getValue(
    #             x
    #         )  # here we compute the value of the contact forces in local frame from world frame
    #         force_msg[i].wrench.force.x = f_local[0]
    #         force_msg[i].wrench.force.y = f_local[1]
    #         force_msg[i].wrench.force.z = f_local[2]
    #         fpubs[i].publish(force_msg[i])
    #
    pub.publish(msg)
    #
    rate.sleep()

roslaunch.kill()
rviz.kill()
