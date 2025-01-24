const std = @import("std");

pub const Tensor = @import("tensor.zig").Tensor;
pub const loss = @import("loss.zig");

pub fn main() !void {
    std.debug.print("Welcome to zinyn, AI in zig using pretty cool tensors\n", .{});
}

test {
    std.testing.refAllDecls(@This());
}
