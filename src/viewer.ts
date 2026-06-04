import {
    type AppBase,
    BoundingBox,
    CameraFrame,
    type CameraComponent,
    Color,
    Entity,
    Layer,
    RenderTarget,
    Mat4,
    MiniStats,
    ShaderChunks,
    PIXELFORMAT_RGBA16F,
    PIXELFORMAT_RGBA32F,
    TONEMAP_NONE,
    TONEMAP_LINEAR,
    TONEMAP_FILMIC,
    TONEMAP_HEJL,
    TONEMAP_ACES,
    TONEMAP_ACES2,
    TONEMAP_NEUTRAL,
    Vec3,
    Vec4,
    GSPLAT_DEBUG_LOD,
    GSPLAT_DEBUG_NONE,
    GSPLAT_RENDERER_RASTER_CPU_SORT,
    GSPLAT_RENDERER_RASTER_GPU_SORT,
    GSplatComponent,
    type ShaderMaterial,
    platform
} from 'playcanvas';

import { Annotations } from './annotations';
import { CameraManager, isWalkAllowed } from './camera-manager';
import { Camera } from './cameras/camera';
import type { Collision } from './collision';
import { MeshCollision, VoxelCollision } from './collision';
import { nearlyEquals } from './core/math';
import { DebugPanel } from './debug';
import { InputController } from './input-controller';
import { MeshDebugOverlay } from './mesh-debug-overlay';
import { NavCursor } from './nav-cursor';
import { Picker } from './picker';
import type { ExperienceSettings, PostEffectSettings } from './settings';
import type { Config, Global } from './types';
import { VoxelDebugOverlay } from './voxel-debug-overlay';

// String.replace wrapper that warns when the source substring is missing, so
// shader chunk patches against the engine fail loudly instead of silently
// producing the original chunk.
const patchChunk = (source: string, search: string, replacement: string, name: string): string => {
    if (!source.includes(search)) {
        console.warn(`patchChunk: substring not found in '${name}', shader chunk patch may be out of sync with the engine.`);
    }
    return source.replace(search, replacement);
};

const gammaChunkGlsl = `
vec3 prepareOutputFromGamma(vec3 gammaColor, float depth) {
    return gammaColor;
}
`;

const gammaChunkWgsl = `
fn prepareOutputFromGamma(gammaColor: vec3f, depth: f32) -> vec3f {
    return gammaColor;
}
`;

const opacityPatchedMaterials = new WeakSet<ShaderMaterial>();

const patchGsplatOpacity = (material: ShaderMaterial, opacity: number, app: AppBase) => {
    if (!opacityPatchedMaterials.has(material)) {
        const glsl = ShaderChunks.get(app.graphicsDevice, 'glsl');
        const wgsl = ShaderChunks.get(app.graphicsDevice, 'wgsl');

        material.shaderChunks.glsl.add({
            gsplatPS: patchChunk(
                patchChunk(
                    glsl.get('gsplatPS'),
                    'uniform float alphaClipForward;',
                    'uniform float alphaClipForward;\nuniform float compareOpacity;',
                    'glsl gsplatPS compare opacity uniform'
                ),
                '#else\n\t\tif (alpha < alphaClipForward) {',
                '#else\n\t\talpha *= compareOpacity;\n\t\tif (alpha < alphaClipForward) {',
                'glsl gsplatPS compare opacity'
            )
        });

        material.shaderChunks.wgsl.add({
            gsplatPS: patchChunk(
                patchChunk(
                    wgsl.get('gsplatPS'),
                    'uniform alphaClipForward: f32;',
                    'uniform alphaClipForward: f32;\nuniform compareOpacity: f32;',
                    'wgsl gsplatPS compare opacity uniform'
                ),
                '#else\n\t\tif (alpha < half(uniform.alphaClipForward)) {',
                '#else\n\t\talpha *= half(uniform.compareOpacity);\n\t\tif (alpha < half(uniform.alphaClipForward)) {',
                'wgsl gsplatPS compare opacity'
            )
        });

        material.update();
        opacityPatchedMaterials.add(material);
    }

    material.setParameter('compareOpacity', opacity);
};

const rendererTable: Record<Config['renderer'], number> = {
    'webgl': GSPLAT_RENDERER_RASTER_CPU_SORT,
    'webgpu': GSPLAT_RENDERER_RASTER_GPU_SORT
};

type GSplatOctreeResourceLike = {
    octree?: {
        lodLevels: number;
    } | null;
};

const tonemapTable: Record<string, number> = {
    none: TONEMAP_NONE,
    linear: TONEMAP_LINEAR,
    filmic: TONEMAP_FILMIC,
    hejl: TONEMAP_HEJL,
    aces: TONEMAP_ACES,
    aces2: TONEMAP_ACES2,
    neutral: TONEMAP_NEUTRAL
};

const applyPostEffectSettings = (cameraFrame: CameraFrame, settings: PostEffectSettings) => {
    if (settings.sharpness.enabled) {
        cameraFrame.rendering.sharpness = settings.sharpness.amount;
    } else {
        cameraFrame.rendering.sharpness = 0;
    }

    const { bloom } = cameraFrame;
    if (settings.bloom.enabled) {
        bloom.intensity = settings.bloom.intensity;
        bloom.blurLevel = settings.bloom.blurLevel;
    } else {
        bloom.intensity = 0;
    }

    const { grading } = cameraFrame;
    if (settings.grading.enabled) {
        grading.enabled = true;
        grading.brightness = settings.grading.brightness;
        grading.contrast = settings.grading.contrast;
        grading.saturation = settings.grading.saturation;
        grading.tint = new Color().fromArray(settings.grading.tint);
    } else {
        grading.enabled = false;
    }

    const { vignette } = cameraFrame;
    if (settings.vignette.enabled) {
        vignette.intensity = settings.vignette.intensity;
        vignette.inner = settings.vignette.inner;
        vignette.outer = settings.vignette.outer;
        vignette.curvature = settings.vignette.curvature;
    } else {
        vignette.intensity = 0;
    }

    const { fringing } = cameraFrame;
    if (settings.fringing.enabled) {
        fringing.intensity = settings.fringing.intensity;
    } else {
        fringing.intensity = 0;
    }
};

const anyPostEffectEnabled = (settings: PostEffectSettings): boolean => {
    return (settings.sharpness.enabled && settings.sharpness.amount > 0) ||
        (settings.bloom.enabled && settings.bloom.intensity > 0) ||
        (settings.grading.enabled) ||
        (settings.vignette.enabled && settings.vignette.intensity > 0) ||
        (settings.fringing.enabled && settings.fringing.intensity > 0);
};

const vec = new Vec3();

// store the original isColorBufferSrgb so the override in updatePostEffects is idempotent
const origIsColorBufferSrgb = RenderTarget.prototype.isColorBufferSrgb;

class Viewer {
    global: Global;

    cameraFrame: CameraFrame;

    inputController: InputController;

    cameraManager: CameraManager;

    picker: Picker;

    annotations: Annotations;

    forceRenderNextFrame = false;

    voxelOverlay: VoxelDebugOverlay | null = null;

    meshOverlay: MeshDebugOverlay | null = null;

    navCursor: NavCursor | null = null;

    debugPanel: DebugPanel | null = null;

    origChunks: {
        glsl: {
            gsplatOutputVS: string,
            skyboxPS: string
        },
        wgsl: {
            gsplatOutputVS: string,
            skyboxPS: string
        }
    };

    constructor(global: Global, gsplatLoads: Promise<Entity>[], skyboxLoad: Promise<void> | undefined, collisionLoad: Promise<Collision> | undefined) {
        this.global = global;

        const { app, settings, config, events, state, camera, renderer } = global;
        const { graphicsDevice } = app;

        // render skybox as plain equirect
        const glsl = ShaderChunks.get(graphicsDevice, 'glsl');
        glsl.set('skyboxPS', patchChunk(glsl.get('skyboxPS'), 'mapRoughnessUv(uv, mipLevel)', 'uv', 'glsl skyboxPS'));

        const wgsl = ShaderChunks.get(graphicsDevice, 'wgsl');
        wgsl.set('skyboxPS', patchChunk(wgsl.get('skyboxPS'), 'mapRoughnessUv(uv, uniform.mipLevel)', 'uv', 'wgsl skyboxPS'));

        this.origChunks = {
            glsl: {
                gsplatOutputVS: glsl.get('gsplatOutputVS'),
                skyboxPS: glsl.get('skyboxPS')
            },
            wgsl: {
                gsplatOutputVS: wgsl.get('gsplatOutputVS'),
                skyboxPS: wgsl.get('skyboxPS')
            }
        };

        // disable auto render, we'll render only when camera changes
        app.autoRender = false;

        // configure the camera
        this.configureCamera(settings);

        // reconfigure camera when entering/exiting XR
        app.xr.on('start', () => this.configureCamera(settings));
        app.xr.on('end', () => this.configureCamera(settings));

        // construct debug ministats
        if (config.ministats) {
            const options = MiniStats.getDefaultOptions() as any;
            options.cpu.enabled = false;
            options.stats = options.stats.filter((s: any) => s.name !== 'DrawCalls');
            options.stats.push({
                name: 'VRAM',
                stats: ['vram.tex'],
                decimalPlaces: 1,
                multiplier: 1 / (1024 * 1024),
                unitsName: 'MB',
                watermark: 1024
            }, {
                name: 'Splats',
                stats: ['frame.gsplats'],
                decimalPlaces: 3,
                multiplier: 1 / 1000000,
                unitsName: 'M',
                watermark: 5
            });

            // eslint-disable-next-line no-new
            new MiniStats(app, options);
        }

        const prevProj = new Mat4();
        const prevWorld = new Mat4();
        const sceneBound = new BoundingBox();
        let wipeCamera: Entity | null = null;
        const enableCameraScissor = (component: CameraComponent, enabled: boolean) => {
            (component as unknown as { _camera: { _scissorRectClear: boolean } })._camera._scissorRectClear = enabled;
        };

        // track the camera state and trigger a render when it changes
        app.on('framerender', () => {
            const world = camera.getWorldTransform();
            const proj = camera.camera.projectionMatrix;

            if (!app.renderNextFrame) {
                if (config.ministats ||
                    !nearlyEquals(world.data, prevWorld.data) ||
                    !nearlyEquals(proj.data, prevProj.data)) {
                    app.renderNextFrame = true;
                }
            }

            // suppress rendering till we're ready
            if (!state.readyToRender) {
                app.renderNextFrame = false;
            }

            if (this.forceRenderNextFrame) {
                app.renderNextFrame = true;
            }

            if (app.renderNextFrame) {
                prevWorld.copy(world);
                prevProj.copy(proj);
            }
        });

        const applyCamera = (camera: Camera) => {
            const cameraEntity = global.camera;

            cameraEntity.setPosition(camera.position);
            cameraEntity.setEulerAngles(camera.angles);
            cameraEntity.camera.fov = camera.fov;

            cameraEntity.camera.horizontalFov = graphicsDevice.width > graphicsDevice.height;

            // fit clipping planes to bounding box
            const boundRadius = sceneBound.halfExtents.length();

            // calculate the forward distance between the camera to the bound center
            vec.sub2(sceneBound.center, camera.position);
            const dist = vec.dot(cameraEntity.forward);

            const far = Math.max(dist + boundRadius, 1e-2);
            const near = Math.max(dist - boundRadius, far / (1024 * 16));

            cameraEntity.camera.farClip = far;
            cameraEntity.camera.nearClip = near;

            if (wipeCamera) {
                wipeCamera.setPosition(camera.position);
                wipeCamera.setEulerAngles(camera.angles);
                wipeCamera.camera.fov = camera.fov;
                wipeCamera.camera.horizontalFov = cameraEntity.camera.horizontalFov;
                wipeCamera.camera.farClip = far;
                wipeCamera.camera.nearClip = near;
                wipeCamera.camera.toneMapping = cameraEntity.camera.toneMapping;
                wipeCamera.camera.clearColor.copy(cameraEntity.camera.clearColor);
            }
        };

        // handle application update
        app.on('update', (deltaTime) => {
            // in xr mode we leave the camera alone
            if (app.xr.active) {
                return;
            }

            if (this.inputController && this.cameraManager) {
                // update inputs
                this.inputController.update(deltaTime, this.cameraManager.camera.distance);

                // update cameras
                this.cameraManager.update(deltaTime, this.inputController.frame);

                // apply to the camera entity
                applyCamera(this.cameraManager.camera);
            }

        });

        // Render voxel debug overlay
        app.on('prerender', () => {
            this.voxelOverlay?.update();
        });

        // update state on first frame
        events.on('firstFrame', () => {
            state.loaded = true;
            state.animationPaused = !!config.noanim;

            window.scrubTo = (time: number) => {
                if (!state.hasAnimation) {
                    return Promise.reject(new Error('No animation track'));
                }

                state.animationPaused = true;
                return new Promise<void>((resolve) => {
                    events.fire('scrubAnim', time);
                    app.renderNextFrame = true;
                    app.once('frameend', () => resolve());
                });
            };

            window.animationDuration = state.animationDuration;
        });

        // wait for the model to load
        Promise.all([Promise.all(gsplatLoads), skyboxLoad, collisionLoad]).then((results) => {
            const gsplatEntities = results[0];
            const collision = results[2];

            // get scene bounding box
            const bounds = gsplatEntities
            .map((entity) => {
                const gsplatComponent = entity.gsplat as GSplatComponent;
                const gsplatBbox = gsplatComponent.customAabb;
                if (!gsplatBbox) {
                    return null;
                }

                const bound = new BoundingBox();
                bound.setFromTransformedAabb(gsplatBbox, entity.getWorldTransform());
                return bound;
            })
            .filter((bound): bound is BoundingBox => !!bound);

            if (bounds.length > 0) {
                sceneBound.copy(bounds[0]);
                for (let i = 1; i < bounds.length; i++) {
                    sceneBound.add(bounds[i]);
                }
            }

            const entityA = gsplatEntities[0];
            const entityB = gsplatEntities[1];
            if (entityB) {
                const gsplatA = entityA.gsplat as GSplatComponent;
                const gsplatB = entityB.gsplat as GSplatComponent;
                const baseLayersA = gsplatA.layers.slice();
                const baseLayersB = gsplatB.layers.slice();
                const compareLayerA = new Layer({ name: 'CompareA' });
                const compareLayerB = new Layer({ name: 'CompareB' });
                const fullRect = new Vec4(0, 0, 1, 1);
                const scissorA = new Vec4();
                const scissorB = new Vec4();
                let blendMaterialA: ShaderMaterial | null = null;
                let blendMaterialB: ShaderMaterial | null = null;

                app.scene.layers.push(compareLayerA);
                app.scene.layers.push(compareLayerB);

                wipeCamera = new Entity('camera wipe b');
                app.root.addChild(wipeCamera);
                wipeCamera.addComponent('camera');
                wipeCamera.camera.enabled = false;
                wipeCamera.camera.priority = camera.camera.priority + 1;

                const withoutCompareLayers = (layers: number[]) => {
                    return layers.filter(id => id !== compareLayerA.id && id !== compareLayerB.id);
                };

                const updateWipeRects = () => {
                    const split = state.wipePosition;
                    scissorA.set(0, 0, split, 1);
                    scissorB.set(split, 0, 1 - split, 1);

                    if (state.compareMode === 'wipe' && wipeCamera) {
                        camera.camera.scissorRect = scissorA;
                        wipeCamera.camera.scissorRect = scissorB;
                        app.renderNextFrame = true;
                    }
                };

                const updateBlendOpacity = () => {
                    const visualBlend = Math.pow(state.blendPosition, config.blendGamma);
                    const opacityA = state.compareMode === 'blend' ? 1 - visualBlend : 1;
                    const opacityB = state.compareMode === 'blend' ? visualBlend : 1;

                    if (blendMaterialA) {
                        patchGsplatOpacity(blendMaterialA, opacityA, app);
                    }
                    if (blendMaterialB) {
                        patchGsplatOpacity(blendMaterialB, opacityB, app);
                    }
                    app.renderNextFrame = true;
                };

                app.systems.gsplat.on('material:created', (material: ShaderMaterial, _camera: unknown, layer: Layer) => {
                    if (layer === compareLayerA) {
                        blendMaterialA = material;
                        updateBlendOpacity();
                    } else if (layer === compareLayerB) {
                        blendMaterialB = material;
                        updateBlendOpacity();
                    }
                });

                const applyCompareMode = () => {
                    const isWipe = state.compareMode === 'wipe';
                    const isBlend = state.compareMode === 'blend';
                    const useCompareLayers = isWipe || isBlend;

                    entityA.enabled = state.compareMode !== 'b';
                    entityB.enabled = state.compareMode !== 'a';
                    gsplatA.layers = useCompareLayers ? [compareLayerA.id] : baseLayersA;
                    gsplatB.layers = useCompareLayers ? [compareLayerB.id] : baseLayersB;
                    camera.camera.rect = fullRect;
                    camera.camera.scissorRect = isWipe ? scissorA : fullRect;
                    enableCameraScissor(camera.camera, isWipe);
                    camera.camera.layers = withoutCompareLayers(camera.camera.layers);
                    if (isWipe) {
                        camera.camera.layers = [...camera.camera.layers, compareLayerA.id];
                    } else if (isBlend) {
                        camera.camera.layers = [...camera.camera.layers, compareLayerA.id, compareLayerB.id];
                    }

                    wipeCamera.camera.enabled = isWipe;
                    wipeCamera.camera.rect = fullRect;
                    wipeCamera.camera.scissorRect = isWipe ? scissorB : fullRect;
                    enableCameraScissor(wipeCamera.camera, isWipe);
                    wipeCamera.camera.layers = isWipe ?
                        [...withoutCompareLayers(camera.camera.layers), compareLayerB.id] :
                        withoutCompareLayers(wipeCamera.camera.layers);

                    updateWipeRects();
                    updateBlendOpacity();
                    app.renderNextFrame = true;
                };

                events.on('compareMode:changed', applyCompareMode);
                events.on('wipePosition:changed', updateWipeRects);
                events.on('blendPosition:changed', updateBlendOpacity);
                applyCompareMode();
            }

            if (!config.noui) {
                this.annotations = new Annotations(global, this.cameraFrame != null);
            }

            this.picker = new Picker(app, camera);
            this.inputController = new InputController(global, this.picker);
            this.inputController.collision = collision ?? null;

            // hasCollision = collision data exists (drives fly-mode collision
            // detection and the voxel/mesh debug overlay availability).
            // walkAllowed = walk mode is offered to the user; requires both
            // collision data and a scene large enough to walk around in.
            state.hasCollision = !!collision;
            state.walkAllowed = isWalkAllowed(sceneBound, collision ?? null);

            // Create collision debug overlay (voxel uses a compute shader, mesh
            // uses standard line rendering). The voxel path requires WebGPU.
            if (collision instanceof VoxelCollision && renderer !== 'webgl') {
                this.voxelOverlay = new VoxelDebugOverlay(app, collision, camera);
                this.voxelOverlay.mode = config.heatmap ? 'heatmap' : 'overlay';
                state.hasCollisionOverlay = true;

                events.on('collisionOverlayEnabled:changed', (value: boolean) => {
                    this.voxelOverlay.enabled = value;
                    app.renderNextFrame = true;
                });
            } else if (collision instanceof MeshCollision) {
                this.meshOverlay = new MeshDebugOverlay(app, collision, camera, !!this.cameraFrame);
                state.hasCollisionOverlay = true;

                events.on('collisionOverlayEnabled:changed', (value: boolean) => {
                    this.meshOverlay.enabled = value;
                    app.renderNextFrame = true;
                });
            }

            this.cameraManager = new CameraManager(global, sceneBound, collision);
            applyCamera(this.cameraManager.camera);

            if (!config.noui) {
                this.navCursor = new NavCursor(app, camera, collision ?? null, events, state);
            }

            this.debugPanel = new DebugPanel(global, this.cameraManager);

            const { gsplat } = app.scene;

            // quality budget
            const budgets = {
                mobile: {
                    low: 1,
                    high: 2
                },
                desktop: {
                    low: 2,
                    high: 4
                }
            };

            const applyPerfSettings = () => {
                const budget = () => {
                    if (config.budget !== undefined && Number.isFinite(config.budget) && config.budget > 0) {
                        return config.budget;
                    }
                    const quality = platform.mobile ? budgets.mobile : budgets.desktop;
                    return state.performanceMode ? quality.low : quality.high;
                };

                gsplat.splatBudget = budget() * 1000000;
                gsplat.lodRangeMin = 0;
                gsplat.lodRangeMax = 1000;
                gsplat.colorUpdateAngle = state.performanceMode ? 4 : 2;
                gsplat.minContribution = 1;
                gsplat.alphaClip = 1 / 255;
                gsplat.antiAlias = config.aa;
            };

            if (config.fullload) {
                // reveal once full quality has finished loading (used for screenshots)
                applyPerfSettings();
            } else {
                // reveal once low lod has loaded for fastest possible reveal
                const resource = gsplatEntities[0].gsplat.resource as GSplatOctreeResourceLike | null;
                const lodLevels = resource?.octree?.lodLevels;
                if (lodLevels) {
                    gsplat.lodRangeMax = gsplat.lodRangeMin = lodLevels - 1;
                }
            }

            // these two allow LOD behind camera to drop, saves lots of splats
            gsplat.lodUpdateAngle = 90;
            gsplat.lodBehindPenalty = 5;

            // same performance, but rotating on slow devices does not give us unsorted splats on sides
            gsplat.radialSorting = true;

            const eventHandler = app.systems.gsplat;

            // idle timer: force continuous rendering until 4s of inactivity
            let idleTime = 0;
            this.forceRenderNextFrame = true;

            app.on('update', (dt: number) => {
                idleTime += dt;
                this.forceRenderNextFrame = idleTime < 4;
            });

            events.on('inputEvent', (type: string) => {
                if (type !== 'interact') {
                    idleTime = 0;
                }
            });

            eventHandler.on('frame:ready', (_camera: CameraComponent, _layer: Layer, ready: boolean, loading: number) => {
                if (loading > 0 || !ready) {
                    idleTime = 0;
                }
            });

            let current = 0;
            let watermark = 1;
            const readyHandler = (camera: CameraComponent, layer: Layer, ready: boolean, loading: number) => {
                if (ready && loading === 0) {
                    // scene is done loading
                    eventHandler.off('frame:ready', readyHandler);

                    state.readyToRender = true;

                    // handle quality mode changes
                    events.on('performanceMode:changed', applyPerfSettings);
                    applyPerfSettings();

                    // debug colorize lods
                    gsplat.debug = config.colorize ? GSPLAT_DEBUG_LOD : GSPLAT_DEBUG_NONE;
                    gsplat.renderer = rendererTable[renderer];

                    // wait for the first valid frame to complete rendering
                    app.once('frameend', () => {
                        events.fire('firstFrame');

                        // emit first frame event on window
                        window.firstFrame?.();
                    });
                }

                // update loading status
                if (loading !== current) {
                    watermark = Math.max(watermark, loading);
                    current = watermark - loading;
                    state.progress = Math.trunc(current / watermark * 100);
                }
            };

            eventHandler.on('frame:ready', readyHandler);
        });
    }

    // configure camera based on application mode and post process settings
    configureCamera(settings: ExperienceSettings) {
        const { global } = this;
        const { app, config, camera } = global;
        const { postEffectSettings } = settings;
        const { background } = settings;

        // hpr override takes precedence over settings.highPrecisionRendering
        const highPrecisionRendering = config.hpr ?? settings.highPrecisionRendering;

        const postFxRequested = !config.nofx &&
            (anyPostEffectEnabled(postEffectSettings) || highPrecisionRendering);

        const enableCameraFrame = !app.xr.active && postFxRequested;

        if (enableCameraFrame) {
            // create instance
            if (!this.cameraFrame) {
                this.cameraFrame = new CameraFrame(app, camera.camera);
            }

            const { cameraFrame } = this;
            cameraFrame.enabled = true;
            cameraFrame.rendering.toneMapping = tonemapTable[settings.tonemapping];
            cameraFrame.rendering.renderFormats = highPrecisionRendering ? [PIXELFORMAT_RGBA16F, PIXELFORMAT_RGBA32F] : [];
            applyPostEffectSettings(cameraFrame, postEffectSettings);
            cameraFrame.update();

            // force gsplat shader to write gamma-space colors
            ShaderChunks.get(app.graphicsDevice, 'glsl').set('gsplatOutputVS', gammaChunkGlsl);
            ShaderChunks.get(app.graphicsDevice, 'wgsl').set('gsplatOutputVS', gammaChunkWgsl);

            // force skybox shader to write gamma-space colors (inline pow replaces the
            // gammaCorrectOutput call which is a no-op under CameraFrame's GAMMA_NONE)
            ShaderChunks.get(app.graphicsDevice, 'glsl').set('skyboxPS',
                patchChunk(
                    this.origChunks.glsl.skyboxPS,
                    'gammaCorrectOutput(toneMap(processEnvironment(linear)))',
                    'pow(toneMap(processEnvironment(linear)) + 0.0000001, vec3(1.0 / 2.2))',
                    'glsl skyboxPS gamma override'
                )
            );
            ShaderChunks.get(app.graphicsDevice, 'wgsl').set('skyboxPS',
                patchChunk(
                    this.origChunks.wgsl.skyboxPS,
                    'gammaCorrectOutput(toneMap(processEnvironment(linear)))',
                    'pow(toneMap(processEnvironment(linear)) + 0.0000001, vec3f(1.0 / 2.2))',
                    'wgsl skyboxPS gamma override'
                )
            );

            // ensure the final compose blit doesn't perform linear->gamma conversion.
            RenderTarget.prototype.isColorBufferSrgb = function (index) {
                return this === app.graphicsDevice.backBuffer ? true : origIsColorBufferSrgb.call(this, index);
            };

            camera.camera.clearColor = new Color(background.color);
        } else {
            // no post effects needed, destroy camera frame if it exists
            if (this.cameraFrame) {
                this.cameraFrame.destroy();
                this.cameraFrame = null;
            }

            // restore shader chunks to engine defaults
            ShaderChunks.get(app.graphicsDevice, 'glsl').set('gsplatOutputVS', this.origChunks.glsl.gsplatOutputVS);
            ShaderChunks.get(app.graphicsDevice, 'wgsl').set('gsplatOutputVS', this.origChunks.wgsl.gsplatOutputVS);
            ShaderChunks.get(app.graphicsDevice, 'glsl').set('skyboxPS', this.origChunks.glsl.skyboxPS);
            ShaderChunks.get(app.graphicsDevice, 'wgsl').set('skyboxPS', this.origChunks.wgsl.skyboxPS);

            // restore original isColorBufferSrgb behavior
            RenderTarget.prototype.isColorBufferSrgb = origIsColorBufferSrgb;

            if (!app.xr.active) {
                camera.camera.toneMapping = tonemapTable[settings.tonemapping];
                camera.camera.clearColor = new Color(background.color);
            }
        }

        // Mesh overlay bakes its vertex colors based on the current gamma
        // path; reapply when CameraFrame is created/destroyed (e.g. on XR
        // start/end) so the overlay tracks the new path.
        this.meshOverlay?.setCameraFrameEnabled(!!this.cameraFrame);
    }
}

export { Viewer };
