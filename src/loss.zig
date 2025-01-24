//! Loss functions tjohei

const std = @import("std");
const testing = std.testing;

const Tensor = @import("tensor.zig").Tensor;

/// Mean squarred error
pub fn mse(lhs: Tensor, rhs: Tensor) f32 {
    _ = lhs;
    _ = rhs;

    return 0.0;
}

pub fn mae(lhs: Tensor, rhs: Tensor) f32 {
    _ = lhs;
    _ = rhs;

    return 0.0;
}

pub fn softmax(lhs: Tensor, rhs: Tensor) f32 {
    _ = lhs;
    _ = rhs;

    return 0.0;
}

// pub fn l1reg(a: Tensor) void {
//     _ = lhs;
//     _ = rhs;
// }
//
// pub fn l2reg(a: Tensor) void {
//     _ = lhs;
//     _ = rhs;
// }
