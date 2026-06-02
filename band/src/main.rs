//! Bevy 0.18 — load a rigged character + drum kit, static stage-lit framing.
//!
//! band/assets/models/character.glb
//! band/assets/models/drums.glb

use bevy::gltf::GltfAssetLabel;
use bevy::prelude::*;

use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::render::render_resource::AsBindGroup;
use bevy::shader::ShaderRef;

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
struct ToonExtension {
    #[uniform(100)]
    quantize_steps: u32,
}
impl Default for ToonExtension {
    fn default() -> Self {
        Self { quantize_steps: 3 }
    } // fewer = flatter/comic-ier
}
impl MaterialExtension for ToonExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/toon.wgsl".into()
    }
    fn deferred_fragment_shader() -> ShaderRef {
        "shaders/toon.wgsl".into()
    }
}

// ---- models ----------------------------------------------------------------
const CHARACTER_PATH: &str = "models/character.glb";
const DRUMS_PATH: &str = "models/drums.glb";

const CHAR_POS: Vec3 = Vec3::new(0.0, 0.0, 0.0);
const CHAR_SCALE: f32 = 1.0;
const CHAR_YAW_DEGREES: f32 = 0.0; // spin the drummer to face the camera if its back is turned

const DRUMS_POS: Vec3 = Vec3::new(0.6, 0.0, 0.0);
const DRUMS_SCALE: f32 = 1.0;
const DRUMS_ROT: f32 = 3.7;

// ---- camera: in front, slightly above, looking down at the drummer ----------
const CAM_POS: Vec3 = Vec3::new(0., 3.3, 9.0);
const CAM_LOOK_AT: Vec3 = Vec3::new(0.0, 2.1, 0.0);
// ----------------------------------------------------------------------------

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(AssetPlugin {
            watch_for_changes_override: Some(true),
            ..default()
        }))
        .add_plugins(bevy_inspector_egui::bevy_egui::EguiPlugin::default())
        .add_plugins(bevy_inspector_egui::quick::WorldInspectorPlugin::new())
        .add_plugins(MaterialPlugin::<ExtendedMaterial<StandardMaterial, ToonExtension>>::default())
        .insert_resource(ClearColor(Color::srgb(1.0, 0.72, 0.62)))
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.5, 0.4, 0.8),
            brightness: 80.0,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(Update, (toonify, aim_spotlights))
        .run();
}

#[derive(Component)]
struct CharacterRoot;

#[derive(Component)]
struct DrumKit;

#[derive(Component)]
struct LightSway {
    speed: f32, // sweep speed
    phase: f32, // offset so beams don't move in lockstep
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Static camera: front, slight top-down.
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(CAM_POS).looking_at(CAM_LOOK_AT, Vec3::Y),
    ));

    // ---- stage lights: blue key, pink fill, purple back-rim, low uplight ----
    // Blue key from front-left, the main shaping light (shadows on).
    commands.spawn((
        SpotLight {
            color: Color::srgb(0.25, 0.45, 1.0),
            intensity: 1_500_000.0,
            range: 30.0,
            inner_angle: 0.75,
            outer_angle: 1.45,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-3.5, 13.0, 8.5),
        LightSway { speed: 0.05, phase: 0.0 },
    ));
    // Pink fill from front-right.
    commands.spawn((
        SpotLight {
            color: Color::srgb(1.0, 0.25, 0.7),
            intensity: 1_500_000.0,
            range: 30.0,
            inner_angle: 0.75,
            outer_angle: 1.45,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(3.5, 8.0, 10.0),
        LightSway { speed: 0.06, phase: 1.7 },
    ));
    // Purple rim from behind/above — separates the figure from the dark stage.
    commands.spawn((
        SpotLight {
            color: Color::srgb(0.6, 0.25, 1.0),
            intensity: 1_500_000.0,
            range: 30.0,
            inner_angle: 0.75,
            outer_angle: 1.45,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(-1.0, 10., 7.0),
        LightSway { speed: 0.04, phase: 3.1 },
    ));
    // Low pink uplight for a bit of concert underglow on the kit.
    commands.spawn((
        SpotLight {
            color: Color::srgb(1.0, 0.3, 0.6),
            intensity: 1_500_000.0,
            range: 30.0,
            inner_angle: 0.75,
            outer_angle: 1.45,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(1.0, 6., 8.0),
        LightSway { speed: 0.07, phase: 4.6 },
    ));

    // ---- models -------------------------------------------------------------
    commands.spawn((
        SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset(CHARACTER_PATH))),
        Transform::from_translation(CHAR_POS)
            .with_scale(Vec3::splat(CHAR_SCALE))
            .with_rotation(Quat::from_rotation_y(CHAR_YAW_DEGREES.to_radians())),
        Name::new("character_root"),
        CharacterRoot,
    ));

    commands.spawn((
        SceneRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset(DRUMS_PATH))),
        Transform::from_translation(DRUMS_POS)
            .with_rotation(Quat::from_axis_angle(Vec3::Y, DRUMS_ROT))
            .with_scale(Vec3::splat(DRUMS_SCALE)),
        Name::new("drum_kit"),
        DrumKit,
    ));

    info!("Static stage framing. Press H to dump bone names.");
}

fn toonify(
    mut commands: Commands,
    q: Query<(Entity, &MeshMaterial3d<StandardMaterial>), Added<MeshMaterial3d<StandardMaterial>>>,
    std_mats: Res<Assets<StandardMaterial>>,
    mut toon: ResMut<Assets<ExtendedMaterial<StandardMaterial, ToonExtension>>>,
) {
    for (e, mat) in &q {
        let Some(base) = std_mats.get(&mat.0) else { continue };
        let handle = toon.add(ExtendedMaterial {
            base: base.clone(),
            extension: ToonExtension::default(),
        });
        commands
            .entity(e)
            .insert(MeshMaterial3d(handle))
            .remove::<MeshMaterial3d<StandardMaterial>>();
    }
}

fn aim_spotlights(time: Res<Time>, mut q: Query<(&LightSway, &mut Transform)>) {
    let t = time.elapsed_secs();
    for (sway, mut tf) in &mut q {
        let target = Vec3::new(
            DRUMS_POS.x + (t * sway.speed + sway.phase).sin() * 3.0,
            DRUMS_POS.y, // ground level of the kit
            DRUMS_POS.z + (t * sway.speed + sway.phase).cos() * 3.0,
        );
        *tf = Transform::from_translation(tf.translation).looking_at(target, Vec3::Y);
    }
}
