//! Benchmark of matmul
const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const allocator = std.heap.page_allocator;

pub fn main() !void {
    const TF32 = Tensor(f32);

    var a = try TF32.fill(allocator, 2.0, &[_]u32{ 2, 2 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 2.0, &[_]u32{ 2, 2 });
    defer b.deinit();

    const iterations = 10;

    const start_time = std.time.nanoTimestamp();

    for (0..iterations) |_| {
        var c = try TF32.mm(&a, &b);
        defer c.deinit();
    }

    const end_time = std.time.nanoTimestamp();

    const elapsed_time = end_time - start_time;
    const average_time_per_iteration = @as(f64, @floatFromInt(elapsed_time)) / @as(f64, @floatFromInt(iterations));

    std.debug.print("Benchmark results:\n", .{});
    std.debug.print("  iterations: {}\n", .{iterations});
    std.debug.print("  elapsed time: {} ns\n", .{elapsed_time});
    std.debug.print("  average time per iteration: {:.2} ns\n", .{average_time_per_iteration});
}
