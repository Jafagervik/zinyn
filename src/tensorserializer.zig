const std = @import("std");
const json = std.json;
const fs = std.fs;

const Tensor = @import("tensor.zig").Tensor;

pub fn TensorSerializer(comptime T: type) type {
    // TODO: Add support for safetensors format
    return struct {
        pub fn saveBinary(self: Tensor(T), path: []const u8) !void {
            const file = try fs.cwd().createFile(path, .{});
            defer file.close();

            // Write metadata
            try file.writeAll(std.mem.asBytes(&@as(u32, self.shape.len)));
            try file.writeAll(std.mem.sliceAsBytes(self.shape));

            try file.writeAll(std.mem.sliceAsBytes(self.data));
        }

        pub fn loadBinary(allocator: std.mem.Allocator, path: []const u8) !Tensor(T) {
            const file = try fs.cwd().openFile(path, .{});
            defer file.close();

            // Read shape metadata
            var shape_len: u32 = undefined;
            _ = try file.read(std.mem.asBytes(&shape_len));

            const shape = try allocator.alloc(u32, shape_len);
            _ = try file.read(std.mem.sliceAsBytes(shape));

            // Calculate total elements
            var total_elements: usize = 1;
            for (shape) |dim| {
                total_elements *= dim;
            }

            const data = try allocator.alloc(T, total_elements);
            _ = try file.read(std.mem.sliceAsBytes(data));

            const strides = try Tensor(T).calculateStrides(allocator, shape);

            return Tensor(T){
                .data = data,
                .shape = shape,
                .strides = strides,
                .allocator = allocator,
            };
        }

        pub fn saveJson(self: Tensor(T), path: []const u8) !void {
            const file = try fs.cwd().createFile(path, .{});
            defer file.close();

            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();

            const writer = file.writer();

            try std.json.stringify(.{
                .dtype = @typeName(T),
                .shape = self.shape,
                .data = self.data,
            }, .{}, writer);
        }

        pub fn loadJson(allocator: std.mem.Allocator, path: []const u8) !Tensor(T) {
            const file = try fs.cwd().openFile(path, .{});
            defer file.close();

            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();

            const reader = file.reader();
            const contents = try reader.readAllAlloc(arena.allocator(), std.math.maxInt(usize));

            const parsed = try std.json.parseFromSlice(std.json.Value, arena.allocator(), contents, .{});
            defer parsed.deinit();

            const root = parsed.value.object;

            // Parse shape
            const json_shape = root.get("shape").?.array;
            var shape = try allocator.alloc(u32, json_shape.items.len);
            for (json_shape.items, 0..) |dim, i| {
                shape[i] = @intCast(dim.integer);
            }

            // Parse data
            const json_data = root.get("data").?.array;
            var data = try allocator.alloc(T, json_data.items.len);
            for (json_data.items, 0..) |val, i| {
                data[i] = switch (T) {
                    f16, f32, f64 => @floatCast(val.float),
                    i16, i32, i64 => @intCast(val.integer),
                    else => @compileError("Unsupported type for JSON serialization"),
                };
            }

            const strides = try Tensor(T).calculateStrides(allocator, shape);

            return Tensor(T){
                .data = data,
                .shape = shape,
                .strides = strides,
                .allocator = allocator,
            };
        }
    };
}
