import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { ClassName } from "../../classes";

// The hero background: the drone photograph laid flat as the ground of a 3D scene, seen from
// almost straight above, with surface layers fitted exactly over its roads. Picking a class
// repaints those roads (wet sheen, standing water, snow, slush, ice) under the scene's own light.

export interface AerialProps {
  /** The surface painted on every road; null shows the photo as taken. */
  surface: ClassName | null;
  reducedMotion: boolean;
  onReady?: () => void;
}

// The photo is 1080 x 1350 px; one world unit is 100 px, centred on the image.
const W = 10.8;
const H = 13.5;
const px = (x: number) => x / 100 - W / 2;
const pz = (y: number) => y / 100 - H / 2;

// Road edges measured on the photo (kerb to kerb).
const ROAD_X: [number, number] = [402, 660]; // the north-south road
const ROAD_Y: [number, number] = [553, 736]; // the east-west road
const CROSSING = { x: px(527), z: pz(643) };

interface OverlayLook {
  color: string;
  /** Light the layer gives off on its own, so snow reads white over a dusk photo. */
  glow?: string;
  opacity: number;
  roughness: number;
  clearcoat: number;
  env: number;
  ripple: number;
}

const LOOK: Record<ClassName, OverlayLook> = {
  dry: { color: "#2a2724", opacity: 0, roughness: 0.95, clearcoat: 0, env: 0.2, ripple: 0 },
  wet: { color: "#0b0f10", opacity: 0.42, roughness: 0.22, clearcoat: 1, env: 1.6, ripple: 0.08 },
  water: { color: "#05090b", opacity: 0.66, roughness: 0.03, clearcoat: 1, env: 2.6, ripple: 0.3 },
  fresh_snow: { color: "#f4f7f8", glow: "#8d9599", opacity: 0.9, roughness: 1, clearcoat: 0, env: 0.5, ripple: 0 },
  melted_snow: { color: "#87939a", glow: "#2c3235", opacity: 0.62, roughness: 0.42, clearcoat: 0.5, env: 1.1, ripple: 0.18 },
  ice: { color: "#b6c8cf", glow: "#2a3a42", opacity: 0.56, roughness: 0.05, clearcoat: 1, env: 2.2, ripple: 0.05 },
};

function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Small interfering wavelets as a normal map, for standing water and slush. */
function rippleTexture(): THREE.DataTexture {
  const size = 256;
  const data = new Uint8Array(size * size * 4);
  const waves = [
    [0.07, 0.03, 0],
    [-0.04, 0.09, 1.3],
    [0.11, -0.06, 2.1],
  ] as const;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let dx = 0;
      let dy = 0;
      for (const [kx, ky, ph] of waves) {
        const c = Math.cos((x * kx + y * ky) * Math.PI + ph);
        dx += kx * c;
        dy += ky * c;
      }
      const n = new THREE.Vector3(-dx * 4, -dy * 4, 1).normalize();
      const i = (y * size + x) * 4;
      data[i] = (n.x * 0.5 + 0.5) * 255;
      data[i + 1] = (n.y * 0.5 + 0.5) * 255;
      data[i + 2] = (n.z * 0.5 + 0.5) * 255;
      data[i + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(data, size, size);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(8, 10);
  tex.anisotropy = 8;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

/** Speckled grain, so snow and slush are not flat paint. */
function grainTexture(): THREE.DataTexture {
  const size = 256;
  const rand = mulberry32(5);
  const data = new Uint8Array(size * size * 4);
  for (let i = 0; i < size * size; i++) {
    const v = (0.82 + 0.18 * rand()) * 255;
    data[i * 4] = data[i * 4 + 1] = data[i * 4 + 2] = v;
    data[i * 4 + 3] = 255;
  }
  const tex = new THREE.DataTexture(data, size, size);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(10, 12);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

/** One soft mask in the shape of the photo's roads: both carriageways, rounded where the kerbs
 * curve into the junction, blurred at the edges so the layer has no seams. */
function roadMask(): THREE.CanvasTexture {
  const scale = 2;
  const w = 1080 * scale;
  const h = 1350 * scale;
  const shape = document.createElement("canvas");
  shape.width = w;
  shape.height = h;
  const ctx = shape.getContext("2d")!;
  ctx.scale(scale, scale);
  ctx.fillStyle = "#fff";
  const [ax, bx] = ROAD_X;
  const [ay, by] = ROAD_Y;
  ctx.fillRect(ax, 0, bx - ax, 1350);
  ctx.fillRect(0, ay, 1080, by - ay);
  // Kerb fillets: the concave curve where each pavement corner rounds into the junction. Fill the
  // corner square, then cut away the circle the kerb follows (it only touches the carriageways).
  const r = 70;
  const corners: [number, number, number, number][] = [
    [ax, ay, -1, -1],
    [bx, ay, 1, -1],
    [ax, by, -1, 1],
    [bx, by, 1, 1],
  ];
  for (const [cx, cy, dx, dy] of corners) {
    ctx.fillRect(Math.min(cx, cx + dx * r), Math.min(cy, cy + dy * r), r, r);
  }
  ctx.globalCompositeOperation = "destination-out";
  for (const [cx, cy, dx, dy] of corners) {
    ctx.beginPath();
    ctx.arc(cx + dx * r, cy + dy * r, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalCompositeOperation = "source-over";
  const soft = document.createElement("canvas");
  soft.width = w;
  soft.height = h;
  const out = soft.getContext("2d")!;
  out.filter = "blur(14px)";
  out.drawImage(shape, 0, 0);
  const tex = new THREE.CanvasTexture(soft);
  tex.anisotropy = 16;
  return tex;
}

const tmp = new THREE.Color();

/** One stretch of road: a rectangle in photo pixels with its own surface layer. */
function RoadLayer({
  rect,
  alpha,
  look,
  reduced,
  ripple,
  grain,
}: {
  rect: [number, number, number, number];
  alpha: THREE.Texture | null;
  look: OverlayLook;
  reduced: boolean;
  ripple: THREE.Texture;
  grain: THREE.Texture;
}) {
  const mat = useRef<THREE.MeshPhysicalMaterial>(null);
  const [x0, y0, x1, y1] = rect;
  useFrame((_, dt) => {
    const m = mat.current;
    if (!m) return;
    const k = reduced ? 1 : 1 - Math.exp(-dt * 4);
    m.color.lerp(tmp.set(look.color), k);
    m.emissive.lerp(tmp.set(look.glow ?? "#000000"), k);
    m.opacity += (look.opacity - m.opacity) * k;
    m.roughness += (look.roughness - m.roughness) * k;
    m.clearcoat += (look.clearcoat - m.clearcoat) * k;
    m.envMapIntensity += (look.env - m.envMapIntensity) * k;
    const r = m.normalScale.x + (look.ripple - m.normalScale.x) * k;
    m.normalScale.set(r, r);
  });
  return (
    <mesh rotation-x={-Math.PI / 2} position={[px((x0 + x1) / 2), 0.004, pz((y0 + y1) / 2)]}>
      <planeGeometry args={[(x1 - x0) / 100, (y1 - y0) / 100]} />
      <meshPhysicalMaterial
        ref={mat}
        color={look.color}
        transparent
        opacity={0}
        depthWrite={false}
        roughness={look.roughness}
        clearcoat={look.clearcoat}
        clearcoatRoughness={0.08}
        normalMap={ripple}
        map={grain}
        alphaMap={alpha}
        envMapIntensity={look.env}
      />
    </mesh>
  );
}

/** A dusk sky and the photo's own street lamps, for the wet and icy surfaces to reflect. */
function Lighting() {
  const { gl, scene } = useThree();
  useEffect(() => {
    const env = new THREE.Scene();
    env.add(
      new THREE.Mesh(
        new THREE.SphereGeometry(50, 32, 16),
        new THREE.ShaderMaterial({
          side: THREE.BackSide,
          vertexShader: "varying vec3 p; void main(){ p = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
          fragmentShader:
            "varying vec3 p; void main(){ float h = normalize(p).y; gl_FragColor = vec4(mix(vec3(0.3,0.27,0.24), vec3(0.55,0.62,0.68), smoothstep(0.0, 0.9, h)), 1.0); }",
        }),
      ),
    );
    const warm = new THREE.MeshBasicMaterial({ color: new THREE.Color(5, 3.6, 2.2) });
    for (let i = 0; i < 10; i++) {
      const lamp = new THREE.Mesh(new THREE.SphereGeometry(1.4, 8, 6), warm);
      const a = (i / 10) * Math.PI * 2;
      const e = 0.6 + (i % 3) * 0.25;
      lamp.position.set(Math.cos(a) * Math.cos(e) * 30, Math.sin(e) * 30, Math.sin(a) * Math.cos(e) * 30);
      env.add(lamp);
    }
    const pmrem = new THREE.PMREMGenerator(gl);
    const rt = pmrem.fromScene(env, 0.03);
    scene.environment = rt.texture;
    return () => {
      scene.environment = null;
      rt.dispose();
      pmrem.dispose();
    };
  }, [gl, scene]);
  // Street lamps where the photo has them.
  const lamps: [number, number][] = [
    [628, 798],
    [1010, 588],
    [180, 612],
  ];
  return (
    <>
      <hemisphereLight args={["#c3cfd8", "#2a2c2b", 1.9]} />
      <directionalLight position={[-4, 10, -6]} intensity={1.6} color="#f2e6d3" />
      {lamps.map(([x, y]) => (
        <pointLight key={`${x}-${y}`} position={[px(x), 0.9, pz(y)]} color="#ffc887" intensity={6} distance={4.5} decay={2} />
      ))}
    </>
  );
}

/** Frames the photo to cover the viewport, crossing right of centre on wide screens. */
function framing(aspect: number, fov: number) {
  const tan = Math.tan(THREE.MathUtils.degToRad(fov / 2));
  let vw: number;
  let cx: number;
  if (aspect >= 1) {
    // Fill the viewport with the road scene without exposing black borders.
    // On wide screens, cover horizontally and vertically.
    vw = Math.min(W * 0.98, H * aspect * 0.92);
    cx = THREE.MathUtils.clamp(CROSSING.x - 0.04 * vw, -W / 2 + vw / 2, W / 2 - vw / 2);
  } else {
    vw = Math.min(W * 0.96, H * aspect * 0.96);
    cx = THREE.MathUtils.clamp(CROSSING.x, -W / 2 + vw / 2, W / 2 - vw / 2);
  }
  const vh = vw / aspect;
  const cz = THREE.MathUtils.clamp(CROSSING.z + 0.04 * vh, -H / 2 + vh / 2, H / 2 - vh / 2);
  return { cx, cz, height: vh / (2 * tan) };
}

function Camera({ reduced, onReady }: { reduced: boolean; onReady?: () => void }) {
  const { camera, size } = useThree();
  const settle = useRef(reduced ? 1 : 0);
  const ready = useRef(false);
  const cam = camera as THREE.PerspectiveCamera;
  useFrame((state, dt) => {
    const f = framing(size.width / size.height, cam.fov);
    if (!reduced) settle.current = Math.min(1, settle.current + dt / 1.8);
    const t = 1 - Math.pow(1 - settle.current, 3);
    const h = f.height * (1.06 - 0.06 * t);
    // A slight lean with the pointer: the flat photo and the road layers stay locked together.
    const lean = reduced ? 0 : 0.08 * t;
    cam.up.set(0, 0, -1);
    cam.position.set(f.cx + state.pointer.x * lean, h, f.cz - state.pointer.y * lean);
    cam.lookAt(f.cx, 0, f.cz);
    if (!ready.current) {
      ready.current = true;
      onReady?.();
    }
  });
  return null;
}

function World({ surface, reducedMotion, onReady }: AerialProps) {
  const photo = useLoader(THREE.TextureLoader, "/home/aerial.jpg");
  photo.colorSpace = THREE.SRGBColorSpace;
  photo.anisotropy = 16;
  photo.generateMipmaps = true;
  photo.minFilter = THREE.LinearMipmapLinearFilter;
  photo.magFilter = THREE.LinearFilter;
  photo.wrapS = THREE.ClampToEdgeWrapping;
  photo.wrapT = THREE.ClampToEdgeWrapping;
  const ripple = useMemo(rippleTexture, []);
  const grain = useMemo(grainTexture, []);
  const mask = useMemo(roadMask, []);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => invalidate(), [surface, invalidate]);
  useFrame((_, dt) => {
    if (!reducedMotion) ripple.offset.y += dt * 0.02;
  });
  const look = LOOK[surface ?? "dry"];
  return (
    <>
      <color attach="background" args={["#050808"]} />
      <Lighting />
      <Camera reduced={reducedMotion} onReady={onReady} />
      {/* Dark asphalt base extension to eliminate any edge seam */}
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.002, 0]}>
        <planeGeometry args={[W * 3, H * 3]} />
        <meshBasicMaterial color="#080b0b" />
      </mesh>
      <mesh rotation-x={-Math.PI / 2}>
        <planeGeometry args={[W, H]} />
        <meshBasicMaterial map={photo} toneMapped={false} />
      </mesh>
      <RoadLayer rect={[0, 0, 1080, 1350]} alpha={mask} look={look} reduced={reducedMotion} ripple={ripple} grain={grain} />
    </>
  );
}

export default function AerialScene(props: AerialProps) {
  return (
    <Canvas
      dpr={[1, 2]}
      frameloop={props.reducedMotion ? "demand" : "always"}
      camera={{ fov: 30, near: 0.1, far: 100, position: [0, 12, 0.001] }}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      onCreated={({ gl }) => {
        gl.toneMapping = THREE.NoToneMapping;
      }}
      aria-hidden="true"
    >
      <World {...props} />
    </Canvas>
  );
}
