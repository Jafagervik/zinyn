const std = @import("std");

pub const Tensor = @import("tensor.zig").Tensor;
pub const SGD = @import("optim.zig").SGD;

const loss = @import("loss.zig");

pub const mse = loss.mse;
pub const mae = loss.mae;
pub const hinge_loss = loss.hinge_loss;
pub const huber_loss = loss.huber_loss;
pub const log_cosh_loss = loss.log_cosh_loss;

test {
    std.testing.refAllDecls(@This());
}
