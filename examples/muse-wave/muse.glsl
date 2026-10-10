// Muse (Jolly) — waving — a signed distance function portrait.
// Muse's default avatar, pink-fur variant, right hand raised in a wave.
// y-up, ~2.5 units tall, front faces +z.
// Paste into a Shadertoy-style raymarcher (map() + standard march loop).
// Fur is left to the viewer's imagination.

float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
    return mix(b, a, h) - k*h*(1.0-h);
}

float sdEllipsoid(vec3 p, vec3 c, vec3 r) {
    vec3 q = (p - c)/r;
    float k0 = length(q), k1 = length(q/r);
    return k0*(k0-1.0)/k1;
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p-a, ba = b-a;
    float h = clamp(dot(pa,ba)/dot(ba,ba), 0.0, 1.0);
    return length(pa-ba*h)-r;
}

float map(vec3 p) {
    float d = sdEllipsoid(p, vec3(0.0,-0.08,0.0), vec3(0.62,0.80,0.55)); // body
    d = smin(d, length(p-vec3(0.0,0.72,0.02))-0.52, 0.28);               // head
    d = min(d, sdEllipsoid(p, vec3(0.0,0.71,0.395),                     // face
                              vec3(0.36,0.36,0.17)));
    d = min(d, length(p-vec3(-0.16,0.80,0.515))-0.065);                 // eyes
    d = min(d, length(p-vec3( 0.16,0.80,0.515))-0.065);
    d = min(d, sdCapsule(p, vec3(-0.12,0.665,0.55),                     // smile
                            vec3(0.0,0.60,0.55), 0.02));
    d = min(d, sdCapsule(p, vec3(0.0,0.60,0.55),
                            vec3(0.12,0.665,0.55), 0.02));
    d = smin(d, sdCapsule(p, vec3(-0.48,0.22,0.0),                      // wave:
                             vec3(-0.70,0.58,0.03), 0.15), 0.12);       //  upper arm
    d = smin(d, sdCapsule(p, vec3(-0.70,0.58,0.03),                     //  forearm
                             vec3(-0.77,0.93,0.06), 0.145), 0.10);
    d = smin(d, length(p-vec3(-0.77,0.97,0.06))-0.155, 0.08);            //  hand
    d = smin(d, sdCapsule(p, vec3(0.50,0.18,0.0),                       // resting arm
                             vec3(0.68,-0.35,0.03), 0.155), 0.12);
    d = smin(d, sdCapsule(p, vec3(-0.24,-0.70,0.0),                      // legs
                             vec3(-0.27,-1.02,0.03), 0.165), 0.12);
    d = smin(d, sdCapsule(p, vec3(0.24,-0.70,0.0),
                             vec3(0.27,-1.02,0.03), 0.165), 0.12);
    return d;
}
