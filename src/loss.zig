//! Loss functions tjohei

const std = @import("std");
const math = std.math;

const Tensor = @import("tensor.zig").Tensor;

/// Mean squared error
pub fn mse(comptime T: type, lhs: *Tensor(T), rhs: *Tensor(T)) T {
    // TODO: Check the difference between tensor shapes
    var res: T = 0;

    for (lhs.data, rhs.data) |l, r| {
        const diff = l - r;
        res += (diff * diff);
    }

    return res / @as(T, @floatFromInt(lhs.size()));
}

/// Mean absolute error
pub fn mae(comptime T: type, lhs: *Tensor(T), rhs: *Tensor(T)) T {
    // TODO: Check the difference between tensor shapes
    var res: T = 0;

    for (lhs.data, rhs.data) |l, r| {
        res += @abs(l - r);
    }

    return res / @as(T, @floatFromInt(lhs.size()));
}

test "mse" {
    const testing = std.testing;
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 4.0, &[_]u32{ 1, 1, 3 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 1, 3 });
    defer b.deinit();

    const ms = mse(f32, &a, &b);

    try testing.expectApproxEqAbs(4.0, ms, 0.001);
}

test "mae" {
    const testing = std.testing;
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 4.0, &[_]u32{ 1, 1, 3 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 1, 3 });
    defer b.deinit();

    const ms = mae(f32, &a, &b);

    try testing.expectEqual(2.0, ms);
}
