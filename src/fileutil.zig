const std = @import("std");

const Tensor = @import("tensor").Tensor;

/// Based on a .safetensor file, get a tensor
pub fn read_tensor_from_file(path: []const u8) !?Tensor {
    _ = path;
}

/// Based on a url, download the safetensor file
pub fn read_tensor_from_url(url: []const u8) !?Tensor {
    var http = std.http.Client{ .allocator = std.heap.page_allocator };
    defer http.deinit();

    const res = http.fetch(.{ .location = .{ .url = url } });
    _ = res;

    // TODO: Parse this result into something meaningful
    return null;
}

/// Common libraries we need to have to make this work
pub fn get_mnist() !?Tensor {
    const MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/";

    return read_tensor_from_url(MNIST_URL);
}
