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

        // NOTE:
        //     These are part of autograd engine

        /// Should grad enabling be required
        requires_grad: bool = false,

        /// The grads themselves, stored in tensor
        grad: ?*Self = null,

        /// Derivative of function
        backward_fn: ?*const fn (*Self, *Self) void = null,

        /// Parent nodes in autograd engine
        parents: [2]?*Self = [_]?*Self{ null, null },

        pub inline fn req_grad(self: *Self) void {
            self.requires_grad = true;
        }

        /// Adds backward pass to compute gradients
        pub fn backward_internal(self: *Self, grad: ?*Tensor) !void {
            // TODO: Check if any of this is actually correct
            if (!self.requires_grad) return;

            if (self.grad == null) {
                // TODO: init
                self.grad = try self.zeros();
            }

            if (self.grad != null) {
                self.grad = try self.grad.add(grad);
            }

            if (self.backward_fn) |f| {
                f(self, self.grad.?);
            }

            // Propagate to parent tensors
            for (self.parents) |parent| {
                parent.backward_internal(self.grad);
            }
        }

        /// Actual backward to be used from this api
        pub fn backward(self: *Self) !void {
            if (self.requires_grad and self.grad == null) {
                self.grad = try Self.init(self.allocator, self.shape);
            }

            if (self.backward_fn) |f| {
                f(self, self.grad.?);
            }

            for (self.parents) |parent| {
                parent.backward();
            }
        }

        inline fn checkShape(lhs: *Self, rhs: *Self) bool {
            return std.mem.eql(u32, lhs.shapeIs(), rhs.shapeIs());
        }

        /// Checks if tensor has gradients enabled
        pub inline fn hasGradient(self: *Tensor) bool {
            return self.requires_grad and self.grad != null;
        }

        /// Init tensor with undefined values
        pub fn init(allocator: Allocator, shape: anytype) !Self {
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
        pub inline fn dtype(self: *Self) type {
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

        /// Get the rank of the tensor
        pub inline fn rank(self: Self) usize {
            return self.shape.len;
        }

        /// NOTE: For me as temp to try stuff
        pub fn setVal(self: *Self, idx: u32, val: T) void {
            if (idx > self.size()) return;

            self.data[idx] = val;
        }

        /// Sets all values in a tensor
        pub inline fn setAll(self: *Self, val: T) void {
            @memset(self.data, val);
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

        /// Get all items
        pub fn to_array(self: *Self) []T {
            return self.data;
        }

        // Get size in bytes
        pub inline fn nbytes(self: *Self) usize {
            return self.data.len * @sizeOf(T);
        }

        /// print the tensor, only stats for now
        pub fn print(self: *Self) void {
            // TODO: Use better printing than debug printing
            std.debug.print("Dtype: {any}\nShape: {any}\nMemory Consumption {d} bytes\n", .{ T, self.shape, self.getMemoryConsumption() });
        }

        // ====================================
        //       ADDITION
        // ====================================

        /// Add two tensors and return a new one
        pub fn add(lhs: *Self, rhs: *Self) !Self {
            if (!lhs.checkShape(rhs)) {
                return error.TensorError;
            }

            var t = try Self.init(lhs.allocator, lhs.shapeIs());

            for (lhs.data, rhs.data, 0..) |l, r, i| {
                t.data[i] = l + r;
            }

            if (lhs.requires_grad or rhs.requires_grad) {
                t.requires_grad = true;
                t.parents = [_]?*Self{ lhs, rhs };

                t.backward_fn = add_backward;
            }

            return t;
        }

        fn add_backward(output: *Self, grad: *Self) void {
            const a = output.parents[0];
            const b = output.parents[1];

            if (a) |at| {
                if (at.requires_grad) {
                    at.backward_internal(grad);
                }
            }

            if (b) |bt| {
                if (bt.requires_grad) {
                    bt.backward_internal(grad);
                }
            }
        }

        pub fn mul(lhs: *Self, rhs: *Self) !Self {
            if (!lhs.checkShape(rhs)) {
                return error.TensorError;
            }

            // Perform forward operation
            var result = try Self.init(lhs.allocator, lhs.shape);
            for (result.data, 0..) |*d, i| {
                d.* = lhs.data[i] * rhs.data[i];
            }

            // Set up autograd metadata
            if (lhs.requires_grad or rhs.requires_grad) {
                result.requires_grad = true;

                result.parents = [_]?*Self{ lhs, rhs };
                // result.parents = try lhs.allocator.alloc(*Self, 2);
                // result.parents[0] = lhs;
                // result.parents[1] = rhs;

                result.backward_fn = mul_backward;
            }

            return result;
        }

        fn mul_backward(output: *Self, grad: *Self) void {
            const a = output.parents[0];
            const b = output.parents[1];

            if (a.?.requires_grad) {
                const grad_a = try Self.init(a.allocator, a.shape);
                for (grad_a.data, 0..) |*d, i| {
                    d.* = grad.data[i] * b.data[i];
                }
                a.backward_internal(&grad_a);
            }

            if (b.?.requires_grad) {
                const grad_b = try Self.init(b.allocator, b.shape);
                for (grad_b.data, 0..) |*d, i| {
                    d.* = grad.data[i] * a.data[i];
                }
                b.backward_internal(&grad_b);
            }
        }
    };
}

test "grads" {
    const allocator = std.testing.allocator;
    const TF32 = Tensor(f32);

    var a = try TF32.fill(allocator, 3.0, &[_]u32{ 2, 2 });
    defer a.deinit();

    var b = try TF32.fill(allocator, 2.0, &[_]u32{ 2, 2 });
    defer b.deinit();

    a.req_grad();
    b.req_grad();

    var c = try a.add(&b);
    defer c.deinit();

    var d = try a.mul(&b);
    defer d.deinit();

    try d.backward();

    std.debug.print("a.grad = {any}\n", .{a.grad.?.data});
    std.debug.print("b.grad = {any}\n", .{b.grad.?.data});

    try testing.expectEqual(3.0, a.getFirst());
}
