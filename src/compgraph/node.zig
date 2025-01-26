//! This structure defines a node to be used in our computational dynamic graph
const std = @import("std");
const testing = std.testing;

pub fn Node(comptime T: type) type {
    return struct {
        const Self = @This();

        value: T,
        grad: T,
        /// Binary operations: add, mul
        op: ?fn (T, T) T = null,
        /// Unary operations: relu, sigmoid
        uop: ?fn (T) T = null,
        inputs: ?[]const Self = null,

        /// Backwards grad
        pub fn backward(self: *Node, grad: T) void {
            self.grad += grad;

            for (self.inputs) |inp| {
                if (self.op) |operation| {
                    const deriv = self.computeDerivative(operation, inp.value);
                    inp.backward(grad * deriv);
                }
            }
        }

        /// Gets derivative of operation
        fn computeUnaryDerivative(operation: fn (T) T, inputVal: T) T {
            return switch (operation) {
                relu => if (inputVal > 0.0) 1.0 else 0.0,
                sigmoid => {
                    const val = sigmoid(inputVal);
                    return val * (1.0 - val);
                },
                else => 0.0,
            };
        }

        /// Gets derivative of operation
        fn computeBinaryDerivative(operation: fn (T, T) T, inputVal: T) T {
            return switch (operation) {
                addition => 1.0,
                multiplication => inputVal,
                else => 0.0,
            };
        }
    };
}

fn relu(x: f64) f64 {
    return if (x > 0.0) x else 0.0;
}

fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

fn tanh(x: f64) f64 {
    return (@exp(x) - @exp(-x)) / (@exp(x) + @exp(-x));
}

fn softmax(x: f64) f64 {
    return x;
}

fn addition(a: f64, b: f64) f64 {
    return a + b;
}

fn multiplication(a: f64, b: f64) f64 {
    return a * b;
}
