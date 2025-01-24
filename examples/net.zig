//! example of net in zig

const std = @import("std");
const Allocator = std.mem.Allocator;
const math = std.math;

const zinyn = @import("zinyn");
const Tensor = zinyn.Tensor;
const mse = zinyn.Tensor.mse;
const Layer = zinyn.Layer;
const Model = zinyn.Model;
const optim = zinyn.optim;
const Adam = optim.Adam;
const mse = zinyn.losses.mse;

fn train_step(x: Tensor(f32), model: *Model, opt: *Adam) Tensor(f32) {
    opt.zero_grad();

    const y = model.forward(x);

    const loss = mse(x, y);

    opt.step();

    return loss;
}

fn main() !void {
    const TF32 = Tensor(f32);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // Load data

    // preprocess

    // set up loss function

    var tensor = try TF32.fill(allocator, math.pi, &[_]usize{ 2, 3, 4 });
    defer tensor.deinit();

    tensor.print();
}
