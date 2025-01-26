const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

/// Enum based interface for optimizers
pub const Optimizer = union(enum) {
    adam: Adam,
    sgd: SGD,

    pub fn step(self: *Optimizer) !void {
        switch (self.*) {
            inline else => |*case| return try case.step(),
        }
    }
};

/// The adam optimizer
pub const Adam = struct {
    learning_rate: f32 = 0.001,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,

    pub fn step(self: *Adam) !void {
        _ = self;
    }
};

/// The SGD optimizer
pub const SGD = struct {
    /// Learning rate for SGD
    learning_rate: f32 = 0.01,

    /// Momentum for SGD
    momentum: f32 = 0.0,

    /// Use nesterov
    nesterov: bool = false,

    velocity: ?[]f32 = null,

    /// Update the gradient descent algo
    pub fn step(self: *SGD, param: *Tensor(f32), grad: *Tensor(f32)) !void {
        for (param.data, grad.data) |*p, g| {
            p.* -= self.learning_rate * g;
        }
    }

    pub fn deinit(self: *SGD) void {
        if (self.velocity) |vel| std.heap.page_allocator.free(vel);
    }
};

test "adam optim" {}
