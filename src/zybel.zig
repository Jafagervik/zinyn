const std = @import("std");

pub const Tensor = @import("tensor.zig").Tensor;
pub const optim = @import("optim.zig");

test {
    std.testing.refAllDecls(@This());
}
