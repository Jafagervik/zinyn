const std = @import("std");
const testing = std.testing;

const Layer = @import("layer.zig").Layer;
const Tensor = @import("tensor.zig").Tensor;

pub const Model = struct {
    const Self = @This();

    layers: []Layer,

    pub fn init(layers: []Layer) Model {
        return .{
            .layers = layers,
        };
    }

    /// Forward layer
    pub fn forward(self: *Self, X: Tensor) Tensor {
        var out: Tensor = X;
        for (self.layers) |layer| {
            out = layer.apply(X);
        }
        return out;
    }
};

test "Model" {
    const allocator = testing.allocator;

    const in = Tensor(f32).ones(allocator, &[_]usize{ 1, 1, 3, 3 });
    defer in.deinit();

    const layers: []Layer = .{
        Conv2d(3, 3),
        Flatten(),
        Linear(10, 1),
    };

    var model = Model.init(layers);

    const out = model.forward(in);
}
