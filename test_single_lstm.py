from __future__ import absolute_import, print_function
from tvm.contrib import cc
from tvm.contrib import util
import tvm
import numpy as np

def single_lstm():
    num_gate = 4
    hidden_size = tvm.var('hidden_size')
    batch_size = tvm.var('batch_size')
    input_size = tvm.var('input_size')

    # A single LSTM block operations without unrolling
    # '*' linear transformation
    # '(*)' elementwise multiplication
    # F_t = sigmoid( W_f * x_t + R_f * h_t-1 + b_f )
    # I_t = sigmoid( W_i * x_t + R_i * h_t-1 + b_i )
    # O_t = sigmoid( W_o * x_t + R_o * h_t-1 + b_o )
    # C'_t = tanh( W_c * x_t + R_c * h_t-1 + b_c )
    # C_t = F_t (*) C_t-1 + I_t (*) C'_t
    # h_t = O_t (*) tanh( C_t )

    # Global transition matrix

    # input X[0..t-1]
    X = tvm.placeholder((batch_size, input_size), name="X")
    Prev_h = tvm.placeholder((batch_size, hidden_size), name="Prev_h")
    Prev_c = tvm.placeholder((batch_size, hidden_size), name="Prev_c")

    # Parameters
    # Weight matrices [W_i, W_f, W_o, W_c]: 4 * hidden_size * input_size
    # Bias: 4 * hidden_size
    Wi2h = tvm.placeholder((num_gate, hidden_size, input_size), name="Wi2h")
    Bi2h = tvm.placeholder((num_gate, hidden_size), name="Bi2h")

    # Weight matrices [R_i, R_f, R_o, R_c]: 4 * hidden_size * hidden_size
    # Only handle hidden transition, saves space.
    Wh2h = tvm.placeholder((num_gate, hidden_size, hidden_size), name="Wh2h")
    Bh2h = tvm.placeholder((num_gate, hidden_size), name="Bh2h")

    # LSTM transition
    # [W_i, W_f, W_o, W_c] * X_t: 4 * num_hidden
    l = tvm.reduce_axis((0, input_size), name="li2h")
    i2h = tvm.compute(
        (batch_size, num_gate, hidden_size),
        lambda i, x, j: tvm.sum(X[i, l] * Wi2h[x, j, l], axis=l),
        name="i2h")

    # [R_i, R_f, R_o, R_c] * h_t-1: 4 * hidden_size
    # R: hidden_size * hidden_size, h: hidden_size * 1
    k = tvm.reduce_axis((0, hidden_size), name="ki2h")
    h2h = tvm.compute(
        (batch_size, num_gate, hidden_size),
        lambda i, x, j: tvm.sum(Prev_h[i, k] * Wh2h[x, j, k], axis=k),
        name="h2h")

    gates = tvm.compute((batch_size, num_gate, hidden_size), 
                        lambda i, j, k: i2h[i, j, k] + h2h[i, j, k] + Bi2h[j, k] + Bh2h[j, k],
                        name="gates")
    gshape = (batch_size, hidden_size)
    in_gate = tvm.compute(gshape, lambda i, j: tvm.sigmoid(gates[i, 0, j]), name="in_gate")
    forget_gate = tvm.compute(gshape, lambda i, j: tvm.sigmoid(gates[i, 1, j]), name="forget_gate")
    out_gate = tvm.compute(gshape, lambda i, j: tvm.sigmoid(gates[i, 2, j]), name="out_gate")
    in_transform = tvm.compute(gshape, lambda i, j: tvm.tanh(gates[i, 3, j]), name="in_transform")

    # C_t = F_t o C_t-1 + I_t o C'_t
    state_c = tvm.compute((batch_size, hidden_size),
                         lambda i, j:
                         forget_gate[i, j] * Prev_c[i, j] +
                         in_gate[i, j] * in_transform[i, j], name="state_c")
    # h_t = O_t o tanh( C_t )
    # state_h = tvm.compute((batch_size, hidden_size), 
    #    lambda i, j: out_gate[i, j] * tvm.tanh(state_c[i, j]), name="state_h")
    out_c, out_h = tvm.compute(
        (batch_size, hidden_size),
        lambda i, j: ( state_c[i, j], out_gate[i, j] * tvm.tanh(state_c[i, j]) ),
        name = "outputs_c_h"	
    	)
    # schedule
    s = tvm.create_schedule(out_h.op)
    print(tvm.lower(s, [X, Prev_h, Prev_c, Wi2h, Bi2h, Wh2h, Bh2h, out_c, out_h], simple_mode=True))
    lstm = tvm.build(s, [X, Prev_h, Prev_c, Wi2h, Bi2h, Wh2h, Bh2h, out_c, out_h], name="single_lstm")
    print(lstm)


    lstm.save("remy_single_lstm.o")
    print(lstm.imported_modules)
    cc.create_shared("remy_single_lstm.so", ["remy_single_lstm.o"])
    


if __name__ == "__main__":
    #single_lstm()
    num_gate = 4
    batch_size = 1
    hidden_size = 2
    input_size = tvm.var('input_size')

    lstm = tvm.module.load("./remy_single_lstm.so")
    

    x_np = np.array([[5]], dtype = 'float32')
    Wi2h_np = np.array([   [ [1], [3] ], [ [-5], [7] ], [ [1], [1] ], [ [1], [1] ]   ], dtype = 'float32')
    Bi2h_np = np.array([   [0 ,0], [0, 0], [0, 0], [0 ,0]     ], dtype = 'float32')
    Wh2h_np = np.array([   [ [0, 0], [0, 0] ], [ [0, 0], [0, 0] ], [ [0, 0], [0, 0] ], [ [0, 0], [0, 0] ]   ], dtype = 'float32')
    Bh2h_np = np.array([   [0 ,0], [0, 0], [0, 0], [0 ,0]     ], dtype = 'float32')
    scan_h_np = np.zeros(shape=(batch_size, hidden_size)).astype("float32")
    scan_c_np = np.zeros(shape=(batch_size, hidden_size)).astype("float32")
    x = tvm.nd.array(x_np)
    Wi2h = tvm.nd.array(Wi2h_np)
    Bi2h = tvm.nd.array(Bi2h_np)
    Wh2h = tvm.nd.array(Wh2h_np)
    Bh2h = tvm.nd.array(Bh2h_np)
    scan_h = tvm.nd.array(scan_h_np)
    scan_c = tvm.nd.array(scan_c_np)
    
    out_c = tvm.nd.array(np.zeros((batch_size, hidden_size)).astype("float32"))
    out_h = tvm.nd.array(np.zeros((batch_size, hidden_size)).astype("float32"))

    lstm(x, scan_h, scan_c, Wi2h, Bi2h, Wh2h, Bh2h, out_c, out_h)
    print(out_h)