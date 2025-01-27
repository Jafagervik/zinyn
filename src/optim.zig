const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;

/// Enum based interface for optimizers
pub const Optimizer = union(enum) {
    // adam: Adam,
    sgd: SGD,

    pub fn step(self: *Optimizer, param: *Tensor(f32), grad: *Tensor(f32)) !void {
        switch (self.*) {
            inline else => |*case| return try case.step(param, grad),
        }
    }
};

/// The adam optimizer
const Adam = struct {
    /// Learning rate for Adam
    learning_rate: f32 = 0.001,

    /// First Beta value for Adam
    beta1: f32 = 0.9,

    /// Second Beta value for Adam
    beta2: f32 = 0.999,

    epsilon: f32 = 1e-8,

    pub fn step(self: *Adam, param: *Tensor(f32), grad: *Tensor(f32)) !void {
        _ = self;
        _ = param;
        _ = grad;
    }
};

/// The SGD optimizer
/// Supports both nesterov and momentum
pub const SGD = struct {
    /// Learning rate for SGD
    learning_rate: f32 = 0.01,

    /// Momentum for SGD
    momentum: f32 = 0.0,

    /// Use nesterov
    nesterov: bool = false,

    /// Velocity used for SGD
    velocity: ?[]f32 = null,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) SGD {
        return .{ .allocator = allocator };
    }

    /// Update the gradient descent algo
    pub fn step(self: *SGD, param: *Tensor(f32), grad: *Tensor(f32)) !void {
        if (self.velocity == null) {
            self.velocity = try self.allocator.alloc(f32, param.size());
            @memset(self.velocity.?, 0);
        }

        for (param.data, grad.data, self.velocity.?) |*p, g, *v| {
            v.* = self.momentum * v.* + g;

            const update = if (self.nesterov) self.momentum * v.* + g else v.*;

            p.* -= self.learning_rate * update;
        }
    }

    pub fn deinit(self: *SGD) void {
        if (self.velocity) |vel| self.allocator.free(vel);
    }
};

test "SGD optimizer" {
    const testing = std.testing;
    const allocator = testing.allocator;
    const TF32 = Tensor(f32);

    var sgd = SGD.init(allocator);
    defer sgd.deinit();

    var param = try TF32.init(allocator, &[_]u32{2});
    defer param.deinit();

    param.setVal(0, 1.0);
    param.setVal(1, 2.0);

    var grad = try TF32.init(allocator, &[_]u32{2});
    defer grad.deinit();

    grad.setVal(0, 0.1);
    grad.setVal(1, 0.2);

    try sgd.step(&param, &grad);

    try testing.expectApproxEqAbs(param.data[0], 0.999, 0.001);
    try testing.expectApproxEqAbs(param.data[1], 1.998, 0.001);
}
