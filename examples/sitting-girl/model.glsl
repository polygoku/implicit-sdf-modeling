vec3 rotX(vec3 q, float a) {
    float c = cos(a), s = sin(a);
    return vec3(q.x, c*q.y - s*q.z, s*q.y + c*q.z);
}
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
    return mix(b, a, h) - k*h*(1.0-h);
}
float sdSphere(vec3 p, vec3 c, float r) { return length(p-c)-r; }
float sdEllipsoid(vec3 p, vec3 c, vec3 r) {
    return (length((p-c)/r)-1.0)*min(r.x,min(r.y,r.z));
}
float sdRoundedBox(vec3 p, vec3 c, vec3 halfExtent, float rounding) {
    vec3 q = abs(p-c) - (halfExtent-vec3(rounding));
    return length(max(q,vec3(0.0))) + min(max(q.x,max(q.y,q.z)),0.0) - rounding;
}
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p-a, ba = b-a;
    float h = clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0);
    return length(pa-ba*h)-r;
}
float sdCylinder(vec3 p, vec3 c, float r, float hh) {
    vec2 d = abs(vec2(length(p.xz-c.xz), p.y-c.y)) - vec2(r, hh);
    return min(max(d.x,d.y),0.0) + length(max(d,vec2(0.0)));
}
float sdRXEllipsoid(vec3 p, vec3 pivot, float ang, vec3 lc, vec3 r) {
    vec3 q = rotX(p-pivot, -ang) - lc;
    return (length(q/r)-1.0)*min(r.x,min(r.y,r.z));
}
float sdRXSphere(vec3 p, vec3 pivot, float ang, vec3 lc, float r) {
    return length(rotX(p-pivot, -ang) - lc) - r;
}
float sdRXTorusArc(vec3 p, vec3 pivot, float ang, vec3 lc,
                   float R, float r, vec2 arc) {
    vec3 q = rotX(p-pivot, -ang) - lc;
    float mid = 0.5*(arc.x+arc.y);
    float a = mid + mod(atan(q.y,q.x)-mid+3.141592653589793,
                        6.283185307179586)-3.141592653589793;
    a = clamp(a,arc.x,arc.y);
    return length(q-vec3(R*cos(a),R*sin(a),0.0))-r;
}

float map(vec3 p) {
    float d = sdEllipsoid(p, vec3(0.00000000, 16.00000000, -6.00000000), vec3(26.00000000, 16.00000000, 22.00000000)); // hips
    d = smin(d, sdCapsule(p, vec3(0.00000000, 22.00000000, -8.00000000), vec3(0.00000000, 60.00000000, -16.00000000), 19.00000000), 5.00000000); // torso
    d = smin(d, sdCapsule(p, vec3(0.00000000, 62.00000000, -15.00000000), vec3(0.00000000, 74.00000000, -13.00000000), 7.00000000), 3.00000000); // neck
    d = smin(d, sdCapsule(p, vec3(0.00000000, 64.00000000, -14.00000000), vec3(0.00000000, 74.00000000, -12.00000000), 9.00000000), 2.00000000); // collar
    d = min(d, sdCapsule(p, vec3(0.00000000, 36.00000000, 7.67000000), vec3(0.00000000, 54.00000000, 3.88000000), 1.60000000)); // zipper
    d = smin(d, sdRXEllipsoid(p, vec3(0.00000000, 90.00000000, -10.00000000), -0.44000000, vec3(0.00000000, 0.00000000, 0.00000000), vec3(19.00000000, 20.50000000, 19.00000000)), 3.00000000); // head
    d = smin(d, sdRXEllipsoid(p, vec3(0.00000000, 90.00000000, -10.00000000), -0.44000000, vec3(0.00000000, 4.00000000, -5.00000000), vec3(20.00000000, 21.50000000, 20.00000000)), 2.00000000); // haircap
    d = min(d, sdRXSphere(p, vec3(0.00000000, 90.00000000, -10.00000000), -0.44000000, vec3(-7.00000000, 3.00000000, 17.50000000), 2.60000000)); // eye_L
    d = min(d, sdRXSphere(p, vec3(0.00000000, 90.00000000, -10.00000000), -0.44000000, vec3(7.00000000, 3.00000000, 17.50000000), 2.60000000)); // eye_R
    d = min(d, sdRXTorusArc(p, vec3(0.00000000, 90.00000000, -10.00000000), -0.44000000, vec3(0.00000000, -3.00000000, 18.00000000), 5.00000000, 1.40000000, vec2(-2.60000000, -0.54000000))); // smile
    d = smin(d, sdCapsule(p, vec3(-15.00000000, 86.00000000, -26.00000000), vec3(-24.00000000, 52.00000000, -31.00000000), 7.50000000), 3.00000000); // ponytail
    d = smin(d, sdSphere(p, vec3(-24.00000000, 50.00000000, -31.00000000), 7.50000000), 2.00000000); // ponytail_end
    d = smin(d, sdCapsule(p, vec3(-11.00000000, 20.00000000, 4.00000000), vec3(-15.00000000, 46.00000000, 24.00000000), 13.00000000), 4.00000000); // thigh_L
    d = smin(d, sdCapsule(p, vec3(11.00000000, 20.00000000, 4.00000000), vec3(15.00000000, 46.00000000, 24.00000000), 13.00000000), 4.00000000); // thigh_R
    d = smin(d, sdSphere(p, vec3(-15.00000000, 48.00000000, 26.00000000), 11.50000000), 3.00000000); // knee_L
    d = smin(d, sdSphere(p, vec3(15.00000000, 48.00000000, 26.00000000), 11.50000000), 3.00000000); // knee_R
    d = smin(d, sdCapsule(p, vec3(-15.00000000, 46.00000000, 26.00000000), vec3(-14.00000000, 10.00000000, 30.00000000), 8.50000000), 3.00000000); // shin_L
    d = smin(d, sdCapsule(p, vec3(15.00000000, 46.00000000, 26.00000000), vec3(14.00000000, 10.00000000, 30.00000000), 8.50000000), 3.00000000); // shin_R
    d = smin(d, sdRoundedBox(p, vec3(-14.00000000, 8.00000000, 40.00000000), vec3(11.00000000, 7.50000000, 14.00000000), 3.50000000), 2.00000000); // shoe_L
    d = smin(d, sdRoundedBox(p, vec3(14.00000000, 8.00000000, 40.00000000), vec3(11.00000000, 7.50000000, 14.00000000), 3.50000000), 2.00000000); // shoe_R
    d = min(d, sdSphere(p, vec3(-21.00000000, 14.00000000, 49.00000000), 2.80000000)); // charm_L1
    d = min(d, sdSphere(p, vec3(-12.00000000, 14.50000000, 51.00000000), 2.40000000)); // charm_L2
    d = min(d, sdSphere(p, vec3(-16.00000000, 15.00000000, 44.00000000), 2.20000000)); // charm_L3
    d = min(d, sdSphere(p, vec3(21.00000000, 14.00000000, 49.00000000), 2.80000000)); // charm_R1
    d = min(d, sdSphere(p, vec3(12.00000000, 14.50000000, 51.00000000), 2.40000000)); // charm_R2
    d = min(d, sdSphere(p, vec3(16.00000000, 15.00000000, 44.00000000), 2.20000000)); // charm_R3
    d = smin(d, sdCapsule(p, vec3(-20.00000000, 56.00000000, -10.00000000), vec3(-27.00000000, 36.00000000, 24.00000000), 8.00000000), 4.00000000); // upperarm_L
    d = smin(d, sdCapsule(p, vec3(20.00000000, 56.00000000, -10.00000000), vec3(27.00000000, 36.00000000, 24.00000000), 8.00000000), 4.00000000); // upperarm_R
    d = smin(d, sdCapsule(p, vec3(-27.00000000, 36.00000000, 24.00000000), vec3(-6.00000000, 29.00000000, 47.00000000), 7.00000000), 4.00000000); // forearm_L
    d = smin(d, sdCapsule(p, vec3(27.00000000, 36.00000000, 24.00000000), vec3(6.00000000, 29.00000000, 47.00000000), 7.00000000), 4.00000000); // forearm_R
    d = smin(d, sdEllipsoid(p, vec3(0.00000000, 28.00000000, 47.00000000), vec3(10.00000000, 7.50000000, 8.50000000)), 3.00000000); // hands
    d = min(d, sdCapsule(p, vec3(17.40000000, 32.80000000, 34.50000000), vec3(24.00000000, 35.00000000, 27.30000000), 7.80000000)); // wristband
    d = min(d, sdCylinder(p, vec3(0.00000000, -1.50000000, 14.00000000), 40.00000000, 2.50000000)); // base
    return d;
}
