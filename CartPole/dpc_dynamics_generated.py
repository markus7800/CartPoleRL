import math
def dpc_dynamics_generated(q_0, q_1, q_2, qdot_0, qdot_1, qdot_2, f, r_1, r_2, m_c, m_1, m_2, g):
	fun=lambda q_0,q_1,q_2,qdot_0,qdot_1,qdot_2,f,r_1,r_2,m_c,m_1,m_2,g:
	([
		qdot_0,
		qdot_1,
		qdot_2,
		(2*f*m_1 - 4*f*m_2*math.cos(2*q_2) + 4*f*m_2 - g*m_1**2*math.sin(2*q_1) - 2*g*m_1*m_2*math.sin(2*q_1) + g*m_1*m_2*math.sin(2*q_1 + 2*q_2) + m_1**2*qdot_1**2*r_1*math.cos(q_1) + 3*m_1*m_2*qdot_1**2*r_1*math.cos(q_1) - m_1*m_2*qdot_1**2*r_1*math.cos(q_1 + 2*q_2) + m_1*m_2*qdot_1**2*r_2*math.cos(q_1 - q_2) + 2*m_1*m_2*qdot_1*qdot_2*r_2*math.cos(q_1 - q_2) + m_1*m_2*qdot_2**2*r_2*math.cos(q_1 - q_2)) /
			(m_1**2*math.cos(2*q_1) + m_1**2 + 2*m_1*m_2*math.cos(2*q_1) - 2*m_1*m_2*math.cos(2*q_2) - m_1*m_2*math.cos(2*q_1 + 2*q_2) + 3*m_1*m_2 + 2*m_1*m_c - 4*m_2*m_c*math.cos(2*q_2) + 4*m_2*m_c),

		(2*m_2*(g*math.cos(q_1 + q_2) + qdot_1**2*r_1*math.sin(q_2))*((2*r_1*math.cos(q_2) + r_2)*(m_1 + m_2 + m_c) - (m_1*r_1*math.sin(q_1) + 2*m_2*r_1*math.sin(q_1) + m_2*r_2*math.sin(q_1 + q_2))*math.sin(q_1 + q_2)) + r_1*(m_1*math.sin(q_1) + 2*m_2*math.sin(q_1) - 2*m_2*math.sin(q_1 + q_2)*math.cos(q_2))*(2*f + m_1*qdot_1**2*r_1*math.cos(q_1) + 2*m_2*qdot_1**2*r_1*math.cos(q_1) + m_2*qdot_1**2*r_2*math.cos(q_1 + q_2) + 2*m_2*qdot_1*qdot_2*r_2*math.cos(q_1 + q_2) + m_2*qdot_2**2*r_2*math.cos(q_1 + q_2)) - 2*(m_1 - m_2*math.sin(q_1 + q_2)**2 + m_2 + m_c)*(g*m_1*r_1*math.cos(q_1) + 2*g*m_2*r_1*math.cos(q_1) + g*m_2*r_2*math.cos(q_1 + q_2) - 2*m_2*qdot_1*qdot_2*r_1*r_2*math.sin(q_2) - m_2*qdot_2**2*r_1*r_2*math.sin(q_2))) /
			(r_1**2*(-m_1**2*math.sin(q_1)**2 + m_1**2 - m_1*m_2*math.sin(q_1)**2 + 2*m_1*m_2*math.sin(q_1)*math.sin(q_2)*math.cos(q_1 + q_2) + 3*m_1*m_2*math.sin(q_2)**2 + m_1*m_2 + m_1*m_c + 4*m_2*m_c*math.sin(q_2)**2)),

		(-2*(g*math.cos(q_1 + q_2) + qdot_1**2*r_1*math.sin(q_2))*((m_1 + m_2 + m_c)*(m_1*r_1**2 + 4*m_2*r_1**2 + 4*m_2*r_1*r_2*math.cos(q_2) + m_2*r_2**2) - (m_1*r_1*math.sin(q_1) + 2*m_2*r_1*math.sin(q_1) + m_2*r_2*math.sin(q_1 + q_2))**2) + 2*((2*r_1*math.cos(q_2) + r_2)*(m_1 + m_2 + m_c) - (m_1*r_1*math.sin(q_1) + 2*m_2*r_1*math.sin(q_1) + m_2*r_2*math.sin(q_1 + q_2))*math.sin(q_1 + q_2))*(g*m_1*r_1*math.cos(q_1) + 2*g*m_2*r_1*math.cos(q_1) + g*m_2*r_2*math.cos(q_1 + q_2) - 2*m_2*qdot_1*qdot_2*r_1*r_2*math.sin(q_2) - m_2*qdot_2**2*r_1*r_2*math.sin(q_2)) - ((2*r_1*math.cos(q_2) + r_2)*(m_1*r_1*math.sin(q_1) + 2*m_2*r_1*math.sin(q_1) + m_2*r_2*math.sin(q_1 + q_2)) - (m_1*r_1**2 + 4*m_2*r_1**2 + 4*m_2*r_1*r_2*math.cos(q_2) + m_2*r_2**2)*math.sin(q_1 + q_2))*(2*f + m_1*qdot_1**2*r_1*math.cos(q_1) + 2*m_2*qdot_1**2*r_1*math.cos(q_1) + m_2*qdot_1**2*r_2*math.cos(q_1 + q_2) + 2*m_2*qdot_1*qdot_2*r_2*math.cos(q_1 + q_2) + m_2*qdot_2**2*r_2*math.cos(q_1 + q_2))) /
		(r_1**2*r_2*(-m_1**2*math.sin(q_1)**2 + m_1**2 - m_1*m_2*math.sin(q_1)**2 + 2*m_1*m_2*math.sin(q_1)*math.sin(q_2)*math.cos(q_1 + q_2) + 3*m_1*m_2*math.sin(q_2)**2 + m_1*m_2 + m_1*m_c + 4*m_2*m_c*math.sin(q_2)**2))])

	return fun(q_0, q_1, q_2, qdot_0, qdot_1, qdot_2, f, r_1, r_2, m_c, m_1, m_2, g)
