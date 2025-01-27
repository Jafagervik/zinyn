//! ==============================================
//!  Everything is a tensor... B-)
//! ==============================================
const std = @import("std");
const math = std.math;
const testing = std.testing;
const Allocator = std.mem.Allocator;

const utils = @import("utils.zig");

/// Tensor data type
pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        shape: []u32,
        strides: []u32,
        allocator: Allocator,

        inline fn checkShape(lhs: Self, rhs: Self) bool {
            return std.mem.eql(u32, lhs.shapeIs(), rhs.shapeIs());
        }

        /// Init tensor with undefined values
        pub fn init(allocator: Allocator, shape: anytype) !Self {
            // TODO: Use tigerstyle?
            var total_elems: usize = 1;

            for (shape) |dim| total_elems *= dim;

            const data = try allocator.alloc(T, total_elems);
            errdefer allocator.free(data);

            const owned_shape = try allocator.dupe(u32, shape);
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

        /// Same as init, but with api closer to pytorch
        pub fn empty(allocator: Allocator, shape: anytype) !Self {
            return Self.init(allocator, shape);
        }

        /// init from other shape
        pub fn zeros_like(allocator: Allocator, other: Self) !Self {
            return Self.zeros(allocator, other.shapeIs());
        }

        /// Init tensor from another one, and fill with ones
        pub fn ones_like(allocator: Allocator, other: Self) !Self {
            return Self.ones(allocator, other.shapeIs());
        }

        /// get shape from tensor
        pub inline fn shapeIs(self: Self) []u32 {
            return self.shape;
        }

        /// Gets the datatype
        pub inline fn dtype(self: Self) type {
            _ = self;
            return T;
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

            const owned_shape = try allocator.dupe(u32, shape);
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

        /// Generate tensor of random numbers between 0 and 1
        pub fn rand(allocator: Allocator, shape: anytype) !Self {
            const t: Self = try Self.init(allocator, shape);

            for (t.data) |*elem| {
                elem.* = try utils.getRandomNumber(null);
            }

            return t;
        }

        pub inline fn randn(allocator: Allocator, lo: T, hi: T, shape: anytype) !Self {
            const t: Self = try Self.init(allocator, shape);

            for (t.data) |*elem| {
                elem.* = try utils.getRandomNumberBetween(lo, hi);
            }

            return t;
        }

        /// Get the rank of the tensor
        pub inline fn rank(self: Self) usize {
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
            // TODO: we dont know if dtype is float here
            var res: T = std.math.floatMax(T);

            for (self.data) |e| {
                if (e < res) {
                    res = e;
                }
            }

            return res;
        }

        /// Get maximum element from tensor
        pub fn mean(self: Self) T {
            // TODO: we dont know if dtype is float here
            var res: f64 = 0;

            for (self.data) |e| {
                res += @floatCast(e);
            }

            res /= @as(f64, self.size());

            return @as(T, res);
        }

        /// Clamps a tensor in place
        pub fn clamp(self: Self, lo: ?T, hi: ?T) void {
            for (self.data) |*e| {
                if (lo) |l| {
                    if (e.* < l) e.* = l;
                }

                if (hi) |h| {
                    if (e.* > h) e.* = h;
                }
            }
        }

        /// NOTE: For me as temp to try stuff
        pub fn setVal(self: *Self, idx: u32, val: T) void {
            if (idx > self.size()) return;

            self.data[idx] = val;
        }

        /// extrema() calls both the min and max of an array
        pub fn extrema(self: Self) [2]T {
            return .{ self.min(), self.max() };
        }

        /// Get first element of tensor
        pub inline fn getFirst(self: *Self) T {
            return self.data[0];
        }

        /// Full size of tensor
        pub inline fn size(self: *Self) u64 {
            return self.data.len;
        }

        fn calculateStrides(allocator: Allocator, shape: []const u32) ![]u32 {
            const strides = try allocator.alloc(u32, shape.len);
            var current_stride: u32 = 1;

            var i = shape.len;
            while (i > 0) : (i -= 1) {
                strides[i - 1] = current_stride;
                current_stride *= shape[i - 1];
            }

            return strides;
        }

        /// get() returns data based on what is passed as a param
        /// A list of numbers gives something
        pub fn get(self: *Self, idx: usize) T {
            return self.data[idx];
        }

        /// Sum of all elements in the tensor
        pub fn sum(self: *Self) T {
            var tot: T = @as(T, 0);

            for (self.data) |e| tot += e;

            return tot;
        }

        /// Product of all elements in the tensor
        pub fn prod(self: *Self) T {
            var tot: T = @as(T, 0);

            for (self.data) |e| tot *= e;

            return tot;
        }

        // Get size in bytes
        inline fn getMemoryConsumption(self: *Self) usize {
            return self.data.len * @sizeOf(T);
        }

        /// print the tensor, only stats for now
        pub fn print(self: *Self) void {
            // TODO: Use better printing than debug printing
            std.debug.print("Dtype: {any}\nShape: {any}\nMemory Consumption {d} bytes\n", .{ T, self.shape, self.getMemoryConsumption() });
        }

        /// reshape the tensor
        pub inline fn reshape(self: *Self, newshape: anytype) !void {
            // For now, reshape only if the len of the two shapes are the same
            // Otherwise just keep the shapes
            if (newshape.len != self.shape.len) return;

            const owned_shape = try self.allocator.dupe(u32, newshape);
            errdefer self.allocator.free(owned_shape);

            self.shape = owned_shape;
        }

        // pub fn transpose(self: *Self, newshape: anytype) void {}

        // ====================================
        //       ADDITION
        // ====================================

        /// Add two tensors and return a new one
        pub fn add(lhs: Self, rhs: Self) !Self {
            // TODO: abstract into method
            if (!std.mem.eql(u32, lhs.shapeIs(), rhs.shapeIs())) {
                return error.TensorError;
            }

            var t = try Self.init(lhs.allocator, lhs.shapeIs());

            for (lhs.data, rhs.data, 0..) |l, r, i| {
                t.data[i] = l + r;
            }

            return t;
        }

        /// Add two tensors and mutate the first one
        pub fn add_mut(lhs: *Self, rhs: Self) !void {
            if (!checkShape(lhs.*, rhs)) {
                return error.TensorError;
            }

            for (rhs.data, 0..) |r, i| {
                lhs.data[i] += r;
            }
        }

        /// Add scalar to tensor and return new tensor
        pub fn add_scalar(self: *Self, value: T) !Self {
            var t = try Self.init(self.allocator, self.shape);

            for (self.data, 0..) |l, i| {
                t.data[i] = l + value;
            }

            return t;
        }

        /// Add scalar to tensor and mutate it
        pub fn add_scalar_mut(self: *Self, value: T) void {
            for (self.data) |*e| {
                e.* += value;
            }
        }

        // ====================================
        //       Subtraction
        // ====================================

        pub fn sub(lhs: Self, rhs: Self) !Self {
            if (!std.mem.eql(u32, lhs.shapeIs(), rhs.shapeIs())) {
                return error.TensorError;
            }

            var t = try Self.init(lhs.allocator, lhs.shapeIs());

            for (lhs.data, rhs.data, 0..) |l, r, i| {
                t.data[i] = l - r;
            }

            return t;
        }

        pub fn sub_mut(lhs: *Self, rhs: Self) !void {
            if (!checkShape(lhs.*, rhs)) {
                return error.TensorError;
            }

            for (rhs.data, 0..) |r, i| {
                lhs.data[i] -= r;
            }
        }

        pub fn sub_scalar(self: *Self, value: T) !Self {
            var t = try Self.init(self.allocator, self.shape);

            for (self.data, 0..) |l, i| {
                t.data[i] = l - value;
            }

            return t;
        }

        pub fn sub_scalar_mut(self: *Self, value: T) void {
            for (self.data) |*e| {
                e.* -= value;
            }
        }

        // ====================================
        //       Multiplication
        // ====================================

        pub fn mul(lhs: Self, rhs: Self) !Self {
            if (!std.mem.eql(u32, lhs.shapeIs(), rhs.shapeIs())) {
                return error.TensorError;
            }

            var t = try Self.init(lhs.allocator, lhs.shapeIs());

            for (lhs.data, rhs.data, 0..) |l, r, i| {
                t.data[i] = l * r;
            }

            return t;
        }

        pub fn mul_mut(lhs: *Self, rhs: Self) !void {
            if (!checkShape(lhs.*, rhs)) {
                return error.TensorError;
            }

            for (rhs.data, 0..) |r, i| {
                lhs.data[i] *= r;
            }
        }

        pub fn mul_scalar(self: *Self, value: T) !Self {
            var t = try Self.init(self.allocator, self.shape);

            for (self.data, 0..) |l, i| {
                t.data[i] = l * value;
            }

            return t;
        }

        pub fn mul_scalar_mut(self: *Self, value: T) void {
            for (self.data) |*e| {
                e.* *= value;
            }
        }

        /// Blocked matrix multiplication between two tensors
        /// of dims {m, n} x {n, p}
        /// MATMUL SHOULD BE FAST AS FUCK
        pub fn matmul(lhs: *Self, rhs: *Self) !Self {
            // Check dims and shape
            if (lhs.shape[1] != rhs.shape[0] or lhs.rank() != 2) {
                return error.TensorError;
            }

            const m = lhs.shape[0];
            const n = lhs.shape[1];
            const p = rhs.shape[1];

            var t = try Self.init(lhs.allocator, &[_]u32{ m, p });

            for (0..m) |i| {
                for (0..p) |k| {
                    var r: T = 0;

                    for (0..n) |j| {
                        const lhs_index = utils.get1DIndex(i, j, n);
                        const rhs_index = utils.get1DIndex(j, k, p);
                        r += lhs.get(lhs_index) * rhs.get(rhs_index);
                    }

                    const tidx: u32 = @intCast(utils.get1DIndex(i, k, p));
                    t.setVal(tidx, r);
                }
            }

            return t;
        }

        /// Shorthand for matmul
        pub fn mm(lhs: *Self, rhs: *Self) !Self {
            return Self.matmul(lhs, rhs);
        }

        // ====================================
        //       Division
        // ====================================

        pub fn div(lhs: Self, rhs: Self) !Self {
            if (!std.mem.eql(u32, lhs.shapeIs(), rhs.shapeIs())) {
                return error.TensorError;
            }

            var t = try Self.init(lhs.allocator, lhs.shapeIs());

            for (lhs.data, rhs.data, 0..) |l, r, i| {
                if (r == 0.0) return error.DivByZeroError;
                t.data[i] = @divExact(l, r);
            }

            return t;
        }

        pub fn div_mut(lhs: *Self, rhs: Self) !void {
            if (!checkShape(lhs.*, rhs)) {
                return error.TensorError;
            }

            for (rhs.data, 0..) |r, i| {
                if (r == @as(T, 0)) return error.DivByZeroError;
                lhs.data[i] /= r;
            }
        }

        pub fn div_scalar(self: *Self, value: T) !Self {
            var t = try Self.init(self.allocator, self.shape);

            for (self.data, 0..) |l, i| {
                t.data[i] = l / value;
            }

            return t;
        }

        pub fn div_scalar_mut(self: *Self, value: T) !void {
            for (self.data) |*e| {
                e.* /= value;
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

        /// TODO: Implement
        pub fn is_approx(self: *Self, other: *Self, eps: f32) bool {
            _ = self;
            _ = other;
            _ = eps;
            return false;
        }

        /// equals requires the same shape as well
        pub fn eq(self: Self, other: Self) bool {
            if (self.size() != other.size() or !self.checkShape(other)) return false;

            var i: usize = 0;
            while (self.data[i]) : (i += 1) {
                if (self.data[i] != other.data[i]) {
                    return false;
                }
            }

            return true;
        }

        /// Squeeze dimensions
        pub fn squeeze(self: *Self, dim: u32) void {
            _ = self;
            _ = dim;
        }

        /// Unqueeze dimensions
        pub fn unsqueeze(self: *Self, dim: u32) void {
            _ = self;
            _ = dim;
        }

        /// math.sqrt
        pub fn sqrt(self: *Self) void {
            for (self.data) |*e| {
                e.* = @sqrt(e);
            }
        }

        /// math.log sets base to 2
        pub fn log(self: *Self) void {
            for (self.data) |*e| {
                e.* = math.log2(e);
            }
        }

        /// math.logn
        pub fn logn(self: *Self, n: u32) void {
            for (self.data) |*e| {
                e.* = math.log(T, n, e);
            }
        }

        /// math.exp
        pub fn exp(self: *Self) void {
            for (self.data) |*e| {
                e.* = @exp(e);
            }
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
            for (self.data) |e| {
                e = if (e > @as(T, 0)) e else 0;
            }
        }

        pub fn relu(self: Self) Self {
            _ = self;
        }

        /// GeLU activation function
        pub fn gelu_mut(self: *Self) void {
            for (self.data) |e| {
                e = if (e > @as(T, 0)) e else e * 0.01;
            }
        }

        pub fn gelu(self: Self) Self {
            _ = self;
        }

        /// sigmoid activation function
        pub fn sigmoid_mut(self: *Self) void {
            for (self.data) |e| {
                e = @divExact(@as(T, 1), @as(T, 1) + math.exp(-e));
            }
        }

        pub fn sigmoid(self: Self) Self {
            _ = self;
        }

        // =============================
        //      Trigonometry
        // =============================

        pub fn sin(self: *Self) void {
            _ = self;
            return void;
        }

        pub fn cos(self: *Self) void {
            _ = self;
            return void;
        }

        pub fn tan(self: *Self) void {
            _ = self;
            return void;
        }

        // =============================
        //      Loss functions
        // =============================

        /// Mean squared error
        pub fn mse(lhs: *Self, rhs: *Self) T {
            // TODO: Check the difference between tensor shapes
            var res: T = 0;

            for (lhs.data, rhs.data) |l, r| {
                const diff = l - r;
                res += (diff * diff);
            }

            return res / @as(T, @floatFromInt(lhs.size()));
        }

        /// Mean absolute error
        pub fn mae(lhs: *Self, rhs: *Self) T {
            // TODO: Check the difference between tensor shapes
            var res: T = 0;

            for (lhs.data, rhs.data) |l, r| {
                res += @abs(l - r);
            }

            return res / @as(T, @floatFromInt(lhs.size()));
        }

        /// Hinge loss
        pub fn hinge_loss(lhs: *Self, rhs: *Self) T {
            // TODO: Check the difference between tensor shapes
            var res: T = 0;

            for (lhs.data, rhs.data) |l, r| {
                res += @max(@as(T, 0), @as(T, 1) - l * r);
            }

            return res / @as(T, @floatFromInt(lhs.size()));
        }

        /// Log-cosh loss
        pub fn log_cosh_loss(lhs: *Self, rhs: *Self) T {
            // TODO: Check the difference between tensor shapes
            var res: T = 0;

            for (lhs.data, rhs.data) |l, r| {
                res += @log(math.cosh(l - r));
            }

            return res / @as(T, @floatFromInt(lhs.size()));
        }

        /// Huber loss
        pub fn huber_loss(lhs: *Self, rhs: *Self, delta: T) T {
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
    };
}

test "Tensor initialization" {
    const TF32 = Tensor(f32);

    const allocator = testing.allocator;
    var tensor = try TF32.init(allocator, &[_]u32{ 2, 3, 4 });
    defer tensor.deinit();

    try testing.expectEqual(tensor.shape.len, 3);
    try testing.expectEqual(tensor.shape[0], 2);
    try testing.expectEqual(tensor.data.len, 24);
}

test "Random tensor init" {
    const TF32 = Tensor(f32);

    const allocator = testing.allocator;
    var tensor = try TF32.ones(allocator, &[_]u32{ 3, 3, 3 });
    defer tensor.deinit();

    const s = tensor.sum();
    try testing.expectEqual(27.0, s);
}

test "minmax extrema" {
    const TF32 = Tensor(f32);

    const allocator = testing.allocator;
    var tensor = try TF32.ones(allocator, &[_]u32{ 3, 3, 3 });
    defer tensor.deinit();

    try testing.expectEqual(.{ 1.0, 1.0 }, tensor.extrema());
}

test "rand" {
    // TODO: Fix faulty test
    const TF32 = Tensor(f32);

    const allocator = testing.allocator;
    var tensor = try TF32.rand(allocator, &[_]u32{ 3, 3, 3 });
    defer tensor.deinit();

    try testing.expect(tensor.min() >= @as(f32, 0));
    try testing.expect(tensor.max() <= @as(f32, 1));
}

test "randn" {
    // TODO: Fix faulty test
    const TF32 = Tensor(f32);

    const allocator = testing.allocator;
    var tensor = try TF32.randn(allocator, 1.0, 4.0, &[_]u32{ 3, 3, 3 });
    defer tensor.deinit();

    try testing.expect(tensor.min() >= @as(f32, 1.0));
    try testing.expect(tensor.max() <= @as(f32, 4.0));
}

test "zeros like" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var tensor = try TF32.init(allocator, &[_]u32{ 3, 3, 3 });
    defer tensor.deinit();

    var zl = try TF32.zeros_like(allocator, tensor);
    defer zl.deinit();

    try testing.expectEqual(@as(f32, 0), zl.getFirst());
}

test "ones like" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var tensor = try TF32.init(allocator, &[_]u32{ 3, 3, 3 });
    defer tensor.deinit();

    var zl = try TF32.ones_like(allocator, tensor);
    defer zl.deinit();

    try testing.expectEqual(@as(f32, 1), zl.getFirst());
}

test "add" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 3.0, &[_]u32{ 1, 3, 4 });
    defer b.deinit();

    var c = try a.add(b);
    defer c.deinit();

    try testing.expectEqual(5.0, c.getFirst());
}

test "add mut" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 3.0, &[_]u32{ 1, 3, 4 });
    defer b.deinit();

    try a.add_mut(b);

    try testing.expectEqual(5.0, a.getFirst());
}

test "add scalar" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    var b = try a.add_scalar(2.0);
    defer b.deinit();

    try testing.expectEqual(4.0, b.getFirst());
}

test "add scalar mut" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    a.add_scalar_mut(5.0);

    try testing.expectEqual(7.0, a.getFirst());
}

test "sub" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 5.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 3.0, &[_]u32{ 1, 3, 4 });
    defer b.deinit();

    var c = try a.sub(b);
    defer c.deinit();

    try testing.expectEqual(2.0, c.getFirst());
}

test "sub mut" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 5.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 3.0, &[_]u32{ 1, 3, 4 });
    defer b.deinit();

    try a.sub_mut(b);

    try testing.expectEqual(2.0, a.getFirst());
}

test "sub scalar" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    var b = try a.sub_scalar(2.0);
    defer b.deinit();

    try testing.expectEqual(0.0, b.getFirst());
}

test "sub scalar mut" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    a.sub_scalar_mut(5.0);

    try testing.expectEqual(-3.0, a.getFirst());
}

test "mul" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 3.0, &[_]u32{ 1, 3, 4 });
    defer b.deinit();

    var c = try a.mul(b);
    defer c.deinit();

    try testing.expectEqual(6.0, c.getFirst());
}

test "mul mut" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 3.0, &[_]u32{ 1, 3, 4 });
    defer b.deinit();

    try a.mul_mut(b);

    try testing.expectEqual(6.0, a.getFirst());
}

test "mul scalar" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    var b = try a.mul_scalar(2.0);
    defer b.deinit();

    try testing.expectEqual(4.0, b.getFirst());
}

test "mul scalar mut" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    a.mul_scalar_mut(5.0);

    try testing.expectEqual(10.0, a.getFirst());
}

test "div" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 6.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 3.0, &[_]u32{ 1, 3, 4 });
    defer b.deinit();

    var c = try a.div(b);
    defer c.deinit();

    try testing.expectEqual(2.0, c.getFirst());
}

test "div mut" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 12.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 3.0, &[_]u32{ 1, 3, 4 });
    defer b.deinit();

    try a.div_mut(b);

    try testing.expectEqual(4.0, a.getFirst());
}

test "div scalar" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    var b = try a.div_scalar(2.0);
    defer b.deinit();

    try testing.expectEqual(1.0, b.getFirst());
}

test "div scalar mut" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 10.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    try a.div_scalar_mut(5.0);

    try testing.expectEqual(2.0, a.getFirst());
}

// test "reshape" {
//     const TF32 = Tensor(f32);
//     const allocator = testing.allocator;
//
//     var a = try TF32.fill(allocator, 10.0, &[_]u32{ 1, 3, 4 });
//     defer a.deinit();
//
//     try a.reshape(&[_]u32{ 1, 6, 2 });
//
//     try testing.expectEqual(&[_]u32{ 1, 6, 2 }, a.shape);
// }

test "clamp" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 10.0, &[_]u32{ 1, 3, 4 });
    defer a.deinit();

    a.clamp(null, 5.0);

    try testing.expectEqual(5.0, a.getFirst());
}

test "mse" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 4.0, &[_]u32{ 1, 1, 3 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 1, 3 });
    defer b.deinit();

    const ms = a.mse(&b);

    try testing.expectApproxEqAbs(4.0, ms, 0.001);
}

test "mae" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 4.0, &[_]u32{ 1, 1, 3 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 1, 3 });
    defer b.deinit();

    const ms = a.mae(&b);

    try testing.expectEqual(2.0, ms);
}

test "hinge_loss" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 1.0, &[_]u32{ 1, 1, 3 });
    defer a.deinit();

    var b = try TF32.fill(allocator, -1.0, &[_]u32{ 1, 1, 3 });
    defer b.deinit();

    const hl = a.hinge_loss(&b);

    try testing.expectEqual(2.0, hl);
}

test "log_cosh_loss" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 4.0, &[_]u32{ 1, 1, 3 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 1, 3 });
    defer b.deinit();

    const lc = a.log_cosh_loss(&b);
    const loss = TF32.log_cosh_loss(&a, &b);

    try testing.expectApproxEqAbs(1.325, lc, 0.001);
    try testing.expectApproxEqAbs(lc, loss, 0.001);
}

test "huber_loss" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 4.0, &[_]u32{ 1, 1, 3 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 2.0, &[_]u32{ 1, 1, 3 });
    defer b.deinit();

    const hl = a.huber_loss(&b, 1.0);

    try testing.expectApproxEqAbs(1.5, hl, 0.001);
}

test "matmul" {
    const TF32 = Tensor(f32);
    const allocator = testing.allocator;

    var a = try TF32.fill(allocator, 2.0, &[_]u32{ 2, 2 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 2.0, &[_]u32{ 2, 2 });
    defer b.deinit();

    var c = try a.matmul(&b);
    defer c.deinit();

    try testing.expectApproxEqAbs(8.0, c.get(0), 0.001);
    try testing.expectApproxEqAbs(8.0, c.get(1), 0.001);
    try testing.expectApproxEqAbs(8.0, c.get(2), 0.001);
    try testing.expectApproxEqAbs(8.0, c.get(3), 0.001);
}
