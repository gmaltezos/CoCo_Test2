integral = 0
prev_error = 0

def PID(model, x0, tau, time_index, inputs_, xss, uss, dss):
    global integral, prev_error

    error = x0[8]-xss[8]

    integral += error*1
    derivative = (error-prev_error)*1
    prev_error = error

    Ki = 0.0
    Kd = 0.005
    Kp = 0.0002


    return max(0, Ki*integral + Kd*derivative + Kp*error + uss), dss 