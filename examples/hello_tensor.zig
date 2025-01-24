const std = @import("std");
const Allocator = std.mem.Allocator;
const math = std.math;

const zinyn = @import("zinyn");
const Tensor = zinyn.Tensor;

fn main() !void {
    const TF32 = Tensor(f32);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var tensor = try TF32.fill(allocator, math.pi, &[_]usize{ 2, 3, 4 });
    defer tensor.deinit();

    tensor.print();
}
