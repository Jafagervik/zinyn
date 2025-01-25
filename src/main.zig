const std = @import("std");

pub const Tensor = @import("tensor.zig").Tensor;
const TF32 = Tensor(f32);
//pub const loss = @import("loss.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var bon: TF32 = try TF32.ones(allocator, &[_]u32{ 1, 3, 3 });
    defer bon.deinit();

    std.debug.print("First value is {d:.3}\n", .{bon.getFirst()});

    bon.setVal(0, 2.0);

    std.debug.print("First value is {d:.3}\n", .{bon.getFirst()});

    std.debug.print("Shape is {any} and dtype is {any} \n", .{ bon.shapeIs(), bon.dtype() });

    std.debug.print("Sum is {d:.3}\n", .{bon.sum()});
}

test {
    std.testing.refAllDecls(@This());
}
