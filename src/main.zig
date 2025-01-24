const std = @import("std");
pub const Tensor = @import("tensor.zig").Tensor;
pub const loss = @import("loss.zig");

pub fn main() !void {
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
}

test {
    std.testing.refAllDecls(@This());
}
