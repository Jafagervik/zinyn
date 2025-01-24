const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;

const utils = @import("utils.zig");

/// Tensor data type
pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();
        pub var dtype: type = T;

        data: []T,
        shape: []usize,
        strides: []usize,
        allocator: Allocator,

        /// Init tensor with undefined values
        pub fn init(allocator: Allocator, shape: anytype) !Self {
            var total_elems: usize = 1;

            for (shape) |dim| total_elems *= dim;

            const data = try allocator.alloc(T, total_elems);
            errdefer allocator.free(data);

            const owned_shape = try allocator.dupe(usize, shape);
            errdefer allocator.free(owned_shape);

            const strides = try calculateStrides(allocator, shape);
            errdefer allocator.free(strides);

            return .{
                .data = data,
                .shape = owned_shape,
                .strides = strides,
                .allocator = allocator,
            };
        }

        /// Deinitialize tensor
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.allocator.free(self.shape);
            self.allocator.free(self.strides);
        }

        /// Similar to init, but uses memset to initialize tensor with default value
        pub fn fill(allocator: Allocator, value: T, shape: anytype) !Self {
            var total_elems: usize = 1;
            for (shape) |dim| total_elems *= dim;

            const data = try allocator.alloc(T, total_elems);
            errdefer allocator.free(data);

            @memset(data, value);

            const owned_shape = try allocator.dupe(usize, shape);
            errdefer allocator.free(owned_shape);

            const strides = try calculateStrides(allocator, shape);
            errdefer allocator.free(strides);

            return .{
                .data = data,
                .shape = owned_shape,
                .strides = strides,
                .allocator = allocator,
            };
        }

        /// Shorthand fill for zeros
        pub fn zeros(allocator: Allocator, shape: anytype) !Self {
            return Self.fill(allocator, @as(T, 0), shape);
        }

        /// Shorthand fill for ones
        pub fn ones(allocator: Allocator, shape: anytype) !Self {
            return Self.fill(allocator, @as(T, 1), shape);
        }

        // ================================
        //    Randoms
        // ================================
        pub fn rand(allocator: Allocator, shape: anytype) !Self {
            const value = try utils.getRandomNumber();
            return Self.fill(allocator, @as(T, value), shape);
        }

        pub fn randn(allocator: Allocator, lo: T, hi: T, shape: anytype) !Self {
            // TODO: Fix
            return Self.fill(allocator, @as(T, lo + hi), shape);
        }

        /// Get the rank of the tensor
        pub fn rank(self: Self) usize {
            return self.shape.len;
        }

        /// Get maximum element from tensor
        pub fn max(self: Self) T {
            var res: T = std.math.floatMin(T);

            for (self.data) |e| {
                if (e > res) {
                    res = e;
                }
            }

            return res;
        }

        /// Get maximum element from tensor
        pub fn min(self: Self) T {
            var res: T = std.math.floatMax(T);

            for (self.data) |e| {
                if (e < res) {
                    res = e;
                }
            }

            return res;
        }

        /// extrema() calls both the min and max of an array
        pub fn extrema(self: Self) [2]T {
            return .{ self.min(), self.max() };
        }

        /// Get first element of tensor
        pub fn getFirst(self: *Self) T {
            return self.data[0];
        }

        /// Full size of tensor
        pub fn size(self: *Self) u64 {
            return self.data.len;
        }

        fn calculateStrides(allocator: Allocator, shape: []const usize) ![]usize {
            const strides = try allocator.alloc(usize, shape.len);
            var current_stride: usize = 1;

            var i = shape.len;
            while (i > 0) : (i -= 1) {
                strides[i - 1] = current_stride;
                current_stride *= shape[i - 1];
            }

            return strides;
        }

        /// Sum of all elements in the tensor
        pub fn sum(self: *Self) T {
            var tot: T = @as(T, 0);

            for (self.data) |e| tot += e;

            return tot;
        }

        // self explanatory
        fn getMemoryConsumption(self: *Self) f32 {
            return self.getFirst() * 4.0;
        }

        /// print the tensor, only stats for now
        pub fn print(self: *Self) void {
            // TODO: Use better printing than debug printing
            std.debug.print("Dtype: {}\n Shape: {}\n: Memory Consumption{}\n", .{ self.dtype, self.shape, self.getMemoryConsumption() });
        }

        /// reshape the tensor
        pub fn reshape(self: *Self, newshape: anytype) void {
            if (newshape.len != self.shape) return;

            self.shape = newshape;
        }

        // pub fn transpose(self: *Self, newshape: anytype) void {}

        // ====================================
        //       ADDITION
        // ====================================

        pub fn element_wise_add(self: Self, other: anytype) Self {
            _ = self;
            _ = other;
        }

        pub fn element_wise_add_mut(self: *Self, other: anytype) void {
            _ = self;
            _ = other;
        }

        pub fn add_scalar(self: *Self, value: T) Self {
            _ = self;
            _ = value;
            return undefined;
        }

        pub fn add_scalar_mut(self: *Self, value: T) void {
            for (self.data) |e| {
                e += value;
            }
        }

        // ====================================
        //       Subtraction
        // ====================================

        pub fn element_wise_sub(self: Self, other: anytype) Self {
            _ = self;
            _ = other;
        }

        pub fn element_wise_sub_mut(self: *Self, other: anytype) void {
            _ = self;
            _ = other;
        }

        pub fn sub_scalar(self: *Self, value: T) Self {
            _ = self;
            _ = value;
            return undefined;
        }

        pub fn sub_scalar_mut(self: *Self, value: T) void {
            for (self.data) |e| {
                e -= value;
            }
        }

        // ====================================
        //       Multiplication
        // ====================================

        pub fn element_wise_mul(self: Self, other: anytype) Self {
            _ = self;
            _ = other;
        }

        pub fn element_wise_mul_mut(self: *Self, other: anytype) void {
            _ = self;
            _ = other;
        }

        pub fn mul_scalar(self: *Self, value: T) Self {
            _ = self;
            _ = value;
            return undefined;
        }

        pub fn mul_scalar_mut(self: *Self, value: T) void {
            for (self.data) |e| {
                e -= value;
            }
        }

        /// MATMUL SHOULD BE FAST AS FUCK
        pub fn matmul(lhs: Self, rhs: Self) void {
            _ = lhs;
            _ = rhs;
        }

        /// shorthand for matmul
        pub fn mm(self: Self, newshape: Self) void {
            self.matmul(newshape);
        }

        // ====================================
        //       Division
        // ====================================

        pub fn element_wise_div(self: Self, other: anytype) !Self {
            _ = self;
            _ = other;
        }

        pub fn element_wise_div_mut(self: *Self, other: anytype) !void {
            _ = self;
            _ = other;
        }

        pub fn div_scalar(self: *Self, value: T) !Self {
            _ = self;
            _ = value;
            return undefined;
        }

        pub fn div_scalar_mut(self: *Self, value: T) !void {
            for (self.data) |e| {
                e /= value;
            }
        }

        // ====================================
        //       Equals
        // ====================================

        /// weak_equals only looks at elements
        pub fn equals(self: Self, other: Self) bool {
            if (self.size() != other.size()) return false;

            var i: usize = 0;
            while (self.data[i]) : (i += 1) {
                if (self.data[i] != other.data[i]) {
                    return false;
                }
            }

            return true;
        }

        /// equals requires the same shape as well
        pub fn isSameAs(self: Self, other: Self) bool {
            if (self.size() != other.size() or self.shape != other.shape) return false;

            var i: usize = 0;
            while (self.data[i]) : (i += 1) {
                if (self.data[i] != other.data[i]) {
                    return false;
                }
            }

            return true;
        }

        /// math.sqrt
        pub fn sqrt(self: *Self) void {
            _ = self;
        }

        /// math.log sets base to 2
        pub fn log(self: *Self) void {
            return self.logn(2);
        }

        /// math.logn
        pub fn logn(self: *Self, n: u32) void {
            _ = self;
            _ = n;
        }

        /// math.exp
        pub fn exp(self: *Self) void {
            _ = self;
        }

        // ==================================
        //  Next
        // ==================================

        //
        // pub fn broadcast_to(self: Self, targetshape: anytype) void {}
        //
        // /// deep copy to new memory location
        // pub fn copy(self: Self) void {}
        //
        // /// create new reference without copying data
        // pub fn view(self: Self) void {}

        // ==================================
        // Activations
        // ==================================

        /// ReLU activation function
        pub fn relu_mut(self: *Self) void {
            _ = self;
        }

        pub fn relu(self: Self) Self {
            _ = self;
        }

        /// GeLU activation function
        pub fn gelu_mut(self: *Self) void {
            _ = self;
        }

        pub fn gelu(self: Self) Self {
            _ = self;
        }

        /// sigmoid activation function
        pub fn sigmoid_mut(self: *Self) void {
            _ = self;
            return 0.0;
        }

        pub fn sigmoid(self: Self) Self {
            _ = self;
        }
    };
}

test "Tensor initialization" {
    const TF32 = Tensor(f32);

    const allocator = testing.allocator;
    var tensor = try TF32.init(allocator, &[_]usize{ 2, 3, 4 });
    defer tensor.deinit();

    try testing.expectEqual(tensor.shape.len, 3);
    try testing.expectEqual(tensor.shape[0], 2);
    try testing.expectEqual(tensor.data.len, 24);
}

test "Random tensor init" {
    const TF32 = Tensor(f32);

    const allocator = testing.allocator;
    var tensor = try TF32.ones(allocator, &[_]usize{ 3, 3, 3 });
    defer tensor.deinit();

    const s = tensor.sum();
    try testing.expectEqual(27.0, s);
}

test "minmax extrema" {
    const TF32 = Tensor(f32);

    const allocator = testing.allocator;
    var tensor = try TF32.ones(allocator, &[_]usize{ 3, 3, 3 });
    defer tensor.deinit();

    try testing.expectEqual(.{ 1.0, 1.0 }, tensor.extrema());
}

test "rand" {
    // TODO: Fix faulty test
    const TF32 = Tensor(f32);

    const allocator = testing.allocator;
    var tensor = try TF32.rand(allocator, &[_]usize{ 3, 3, 3 });
    defer tensor.deinit();

    try testing.expectEqual(27, tensor.size());
}

test "randn" {
    // TODO: Fix faulty test
    const TF32 = Tensor(f32);

    const allocator = testing.allocator;
    var tensor = try TF32.randn(allocator, 1.0, 4.0, &[_]usize{ 3, 3, 3 });
    defer tensor.deinit();

    try testing.expectEqual(27, tensor.size());
}

// TODO: Fill in these tests
test "add" {}
test "sub" {}
test "mul" {}
test "matmul" {}
test "div" {}
