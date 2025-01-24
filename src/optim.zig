const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

/// Enum based interface for optimizers
pub const Optimizer = union(enum) {
    adam: Adam,
    sgd: SGD,

    pub fn init(self: *Optimizer) !Optimizer {
        switch (self.*) {
            inline else => |*case| return try case.init(),
        }
    }

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

    pub fn init(learning_rate: ?f32, beta1: ?f32, beta2: ?f32, epsilon: ?f32) Adam {
        return .{
            .learning_rate = if (learning_rate) |lr| lr else 0.001,
            .beta1 = if (beta1) |b| b else 0.9,
            .beta2 = if (beta2) |b| b else 0.999,
            .epsilon = if (epsilon) |eps| eps else 1e-8,
        };
    }

    /// Use all default params
    pub fn default() Adam {
        return Adam{};
    }

    pub fn step(self: *Adam) !void {
        _ = self;
    }
};

/// The SGD optimizer
pub const SGD = struct {
    learning_rate: f32,
    momentum: f32,
    nesterov: bool,

    velocity: ?[]f32 = null,

    pub fn init(learning_rate: f32, momentum: f32, nesterov: bool) SGD {
        return .{
            .learning_rate = learning_rate,
            .momentum = momentum,
            .nesterov = nesterov,
        };
    }

    /// Update the gradient descent algo
    pub fn step(
        self: *SGD,
        param: *Tensor(f32),
        grad: *Tensor(f32),
    ) !void {
        for (param.data, grad.data) |*p, g| {
            p.* -= self.learning_rate * g;
        }
    }

    pub fn deinit(self: *SGD) void {
        if (self.velocity) |vel| std.heap.page_allocator.free(vel);
    }
};

test "adam optim" {
    const optim = Adam.default();
    _ = optim;
}
