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

/// Hinge loss
pub fn hinge_loss(comptime T: type, lhs: *Tensor(T), rhs: *Tensor(T)) T {
    // TODO: Check the difference between tensor shapes
    var res: T = 0;

    for (lhs.data, rhs.data) |l, r| {
        res += @max(@as(T, 0), @as(T, 1) - l * r);
    }

    return res / @as(T, @floatFromInt(lhs.size()));
}

/// Log-cosh loss
pub fn log_cosh_loss(comptime T: type, lhs: *Tensor(T), rhs: *Tensor(T)) T {
    // TODO: Check the difference between tensor shapes
    var res: T = 0;

    for (lhs.data, rhs.data) |l, r| {
        res += @log(math.cosh(l - r));
    }

    return res / @as(T, @floatFromInt(lhs.size()));
}

/// Huber loss
pub fn huber_loss(comptime T: type, lhs: *Tensor(T), rhs: *Tensor(T), delta: T) T {
    // TODO: Check the difference between tensor shapes
    var res: T = 0;

    for (lhs.data, rhs.data) |l, r| {
        const diff = l - r;
        res += if (@abs(diff) <= delta)
            (diff * diff) / @as(T, 2)
        else
            delta * (@abs(diff) - delta / @as(T, 2));
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

test "hinge_loss" {
    const testing = std.testing;
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 1.0, &[_]u32{ 1, 1, 3 });
    defer a.deinit();

    var b = try TF32.fill(allocator, -1.0, &[_]u32{ 1, 1, 3 });
    defer b.deinit();

    const hl = hinge_loss(f32, &a, &b);

    try testing.expectEqual(2.0, hl);
}

test "log_cosh_loss" {
    const testing = std.testing;
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 4.0, &[_]u32{ 1, 1, 3 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 1, 3 });
    defer b.deinit();

    const lc = log_cosh_loss(f32, &a, &b);

    try testing.expectApproxEqAbs(1.325, lc, 0.001);
}

test "huber_loss" {
    const testing = std.testing;
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 4.0, &[_]u32{ 1, 1, 3 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 1, 3 });
    defer b.deinit();

    const hl = huber_loss(f32, &a, &b, 1.0);

    try testing.expectApproxEqAbs(1.5, hl, 0.001);
}
