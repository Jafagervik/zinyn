const std = @import("std");

pub fn getRandomNumber(seed: ?u64) !f32 {
    var sd: u64 = undefined;

    // Let user be able to set seed manually
    if (seed) |s| {
        sd = s;
    }

    std.posix.getrandom(std.mem.asBytes(&sd)) catch |err| {
        std.debug.print("Failed to get random seed: {}\n", .{err});
        return error.RandSeedError;
    };

    // Needed in 0.14.234523454 dev something
    var prng = std.Random.DefaultPrng.init(sd);
    const rnd = prng.random();

    // Generate a random float between 0 and 1
    return rnd.float(f32);
}

pub fn getRandomNumberBetween(lo: f32, hi: f32) !f32 {
    const random_number = try getRandomNumber(null);
    return lo + (random_number * (hi - lo));
}
