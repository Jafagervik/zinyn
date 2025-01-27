const std = @import("std");
pub const Tensor = @import("tensor.zig").Tensor;
pub const SGD = @import("optim.zig").SGD;
pub const loss = @import("loss.zig");

test {
    std.testing.refAllDecls(@This());
}
