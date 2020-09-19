import math
def spc_dynamics_generated(q_0, q_1, qdot_0, qdot_1, f, r_1, m_c, m_1, g):
	fun=lambda q_0,q_1,qdot_0,qdot_1,f,r_1,m_c,m_1,g:
	([
		qdot_0,
		qdot_1,
		(f - 1/2*g*m_1*math.sin(2*q_1) + (1/2)*m_1*qdot_1**2*r_1*math.cos(q_1))/(m_1*math.cos(q_1)**2 + m_c),
		(-2*g*(m_1 + m_c)*math.cos(q_1) + (2*f + m_1*qdot_1**2*r_1*math.cos(q_1))*math.sin(q_1))/(r_1*(m_1*math.cos(q_1)**2 + m_c))])
	return fun(q_0, q_1, qdot_0, qdot_1, f, r_1, m_c, m_1, g)
