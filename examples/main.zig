const std = @import("std");

pub const Node = @import("compgraph/node.zig").Node;
pub const Tensor = @import("tensor.zig").Tensor;
const TF32 = Tensor(f32);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var bon: TF32 = try TF32.rand(allocator, &[_]u32{ 1, 3, 3 });
    defer bon.deinit();

    for (bon.data) |e| {
        std.log.info("{d:.2}", .{e});
    }

    std.log.info("extrema {any}", .{bon.extrema()});

    bon.print();
}

test {
    std.testing.refAllDecls(@This());
}
