const std = @import("std");

pub fn getRandomNumber() !f32 {
    var seed: u64 = undefined;
    std.posix.getrandom(std.mem.asBytes(&seed)) catch |err| {
        std.debug.print("Failed to get random seed: {}\n", .{err});
        return error.RandSeedError;
    };

    // Needed in 0.14.234523454 dev something
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();

    // Generate a random float between 0 and 1
    return rnd.float(f32);
}

pub fn getRandomNumberBetween(lo: f32, hi: f32) !f32 {
    var seed: u64 = undefined;
    std.posix.getrandom(std.mem.asBytes(&seed)) catch |err| {
        std.debug.print("Failed to get random seed: {}\n", .{err});
        return error.RandSeedError;
    };

    // Needed in 0.14.234523454 dev something
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();

    return lo + (rnd.float(f32) * (hi - lo));
}
