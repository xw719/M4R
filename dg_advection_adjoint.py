from firedrake import *
from firedrake_adjoint import *
from firedrake.petsc import PETSc
import math



def run_forward():

    # define mesh
    n = 600.
    mesh = UnitSquareMesh(40, 40, quadrilateral=True)

    # Define P2 function space and corresponding test function
    V = FunctionSpace(mesh, "DQ", 1)
    W = VectorFunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)

    # Assign initial conditions
    x, y = SpatialCoordinate(mesh)
    velocity = as_vector((0.5 - y, x - 0.5))
    u = interpolate(velocity, W)

    bell_r0, bell_x0, bell_y0  = 0.15, 0.25, 0.5
    cone_r0, cone_x0, cone_y0 = 0.15, 0.5, 0.25
    cyl_r0, cyl_x0, cyl_y0 = 0.15, 0.5, 0.75
    slot_left, slot_right, slot_top = 0.475, 0.525, 0.85

    bell = 0.25*(1+cos(math.pi*min_value(sqrt(pow(x-bell_x0, 2) + pow(y-bell_y0, 2))/bell_r0, 1.0)))
    cone = 1.0 - min_value(sqrt(pow(x-cone_x0, 2) + pow(y-cone_y0, 2))/cyl_r0, 1.0)
    slot_cyl = conditional(sqrt(pow(x-cyl_x0, 2) + pow(y-cyl_y0, 2)) < cyl_r0,
                conditional(And(And(x > slot_left, x < slot_right), y < slot_top),
                0.0, 1.0), 0.0)
    
    # Create Functions for the solution and time-lagged solution
    ic = project(1.0 + bell + cone + slot_cyl,  V)
    q = Function(V).assign(ic)
    # qs = []

    # Define linear form
    T = 2*math.pi
    dt = T/n
    dtc = Constant(dt)
    q_in = Constant(1.0)

    dq_trial = TrialFunction(V)
    phi = TestFunction(V)
    a = phi*dq_trial*dx

    n = FacetNormal(mesh)
    un = 0.5*(dot(u, n) + abs(dot(u, n)))

    # compute right hand side
    L1 = dtc*(q*div(phi*u)*dx
          - conditional(dot(u, n) < 0, phi*dot(u, n)*q_in, 0.0)*ds
          - conditional(dot(u, n) > 0, phi*dot(u, n)*q, 0.0)*ds
          - (phi('+') - phi('-'))*(un('+')*q('+') - un('-')*q('-'))*dS)
    
    q1 = Function(V); q2 = Function(V)
    L2 = replace(L1, {q: q1}); L3 = replace(L1, {q: q2})

    # define variable to hold temporary variables
    dq = Function(V)

    params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
    prob1 = LinearVariationalProblem(a, L1, dq)
    solv1 = LinearVariationalSolver(prob1, solver_parameters=params)
    prob2 = LinearVariationalProblem(a, L2, dq)
    solv2 = LinearVariationalSolver(prob2, solver_parameters=params)
    prob3 = LinearVariationalProblem(a, L3, dq)
    solv3 = LinearVariationalSolver(prob3, solver_parameters=params)

    t = 0.0
    step = 0
    while t < T - 0.5*dt:
        solv1.solve()
        q1.assign(q + dq)

        solv2.solve()
        q2.assign(0.75*q + 0.25*(q1 + dq))

        solv3.solve()
        q.assign((1.0/3.0)*q + (2.0/3.0)*(q2 + dq))

        step += 1
        t += dt

    # calulate objective functional
    J=(assemble((q - ic)*(q - ic)*dx))**0.5

    return ReducedFunctional(J, Control(ic)), ic

with PETSc.Log.Event("Tape_forward"):
    Jhat, ic = run_forward()

with PETSc.Log.Event("Compute_gradient"):
    Jhat.derivative()

with PETSc.Log.Event("Replay"):
    Jhat(ic)

tape = get_working_tape()
from progress.bar import FillingSquaresBar
tape.progress_bar = lambda *args: FillingSquaresBar(*args, suffix='block %(index)d of %(max)d')