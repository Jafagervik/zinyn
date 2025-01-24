pub const Layer = struct {};

pub const Model = struct {
    layers: []Layer,

    pub fn init(layers: []Layer) Model {
        return .{
            .layers = layers,
        };
    }
};
