// dx[0]/dλ = -p_t*r/(r - rs);
// dx[1]/dλ = p_r*(r - rs)/r;
// dx[2]/dλ = p_theta/pow(r, 2);
// dx[3]/dλ = p_phi/(pow(r, 2)*pow(sin(theta), 2));
// ----
// dp[0]/dλ = 0;
// dp[1]/dλ = pow(p_phi, 2)/(pow(r, 3)*pow(sin(theta), 2)) - 1.0/2.0*pow(p_r, 2)/r + (1.0/2.0)*pow(p_r, 2)*(r - rs)/pow(r, 2) - 1.0/2.0*pow(p_t, 2)*r/pow(r - rs, 2) + (1.0/2.0)*pow(p_t, 2)/(r - rs) + pow(p_theta, 2)/pow(r, 3);
// dp[2]/dλ = pow(p_phi, 2)*cos(theta)/(pow(r, 2)*pow(sin(theta), 3));
// dp[3]/dλ = 0;

// vec4(x=t, y=r, z=theta, w=phi)

#define PI 3.1415926535897932384626433832795

vec4 dxdlambda(vec4 x, vec4 p, float rs) {
    float x_t = x.x;
    float x_r = x.y;
    float x_theta = x.z;
    float x_phi = x.w;
    
    float p_t = p.x;
    float p_r = p.y;
    float p_theta = p.z;
    float p_phi = p.w;
    
    float sin_x_theta = sin(x_theta);
    float cos_x_theta = cos(x_theta);

    vec4 dx = vec4(0.);
    
    dx.x = -p_t * x_r / (x_r - rs);                     // -p_t * r / (r - rs)
    dx.y = p_r * (x_r - rs) / x_r;                      // p_r * (r - rs) / r
    dx.z = p_theta / (x_r * x_r);                       // p_theta / pow(r, 2)
    dx.w = p_phi / (x_r*x_r * sin_x_theta*sin_x_theta); // p_phi / (pow(r, 2) * pow(sin(theta), 2))
    
    return dx;
}

vec4 dpdlambda(vec4 x, vec4 p, float rs) {
    float x_t = x.x;
    float x_r = x.y;
    float x_theta = x.z;
    float x_phi = x.w;
    
    float p_t = p.x;
    float p_r = p.y;
    float p_theta = p.z;
    float p_phi = p.w;
    
    float sin_x_theta = sin(x_theta);
    float cos_x_theta = cos(x_theta);

    vec4 dp = vec4(0.);

    // 0;
    dp.x = 0.;
    
    // pow(p_phi, 2) / (pow(r, 3) * pow(sin(theta), 2))
    // - 1.0/2.0 * pow(p_r, 2) / r
    // + (1.0/2.0) * pow(p_r, 2) * (r - rs) / pow(r, 2)
    // - 1.0/2.0 * pow(p_t, 2) * r / pow(r - rs, 2)
    // + (1.0/2.0) * pow(p_t, 2) / (r - rs)
    // + pow(p_theta, 2) / pow(r, 3);
    dp.y = p_phi*p_phi / (x_r*x_r*x_r * sin_x_theta*sin_x_theta)
        - 0.5 * p_r*p_r / x_r
        + 0.5 * p_r*p_r * (x_r - rs) / (x_r*x_r)
        - 0.5 * p_t*p_t * x_r / ((x_r - rs)*(x_r - rs))
        + 0.5 * p_t*p_t / (x_r - rs)
        + p_theta*p_theta / (x_r*x_r*x_r);
    
    // pow(p_phi, 2) * cos(theta) / (pow(r, 2) * pow(sin(theta), 3));
    dp.z = p_phi*p_phi * cos_x_theta / (x_r*x_r * sin_x_theta*sin_x_theta*sin_x_theta);

    // 0;
    dp.w = 0.;

    return dp;
}

void runge_kutta_step(inout vec4 x, inout vec4 p, float rs, float lambda_step) {
    vec4 k1_x = dxdlambda(x, p, rs);
    vec4 k1_p = dpdlambda(x, p, rs);
    vec4 k2_x = dxdlambda(x + 0.5*lambda_step*k1_x, p + 0.5*lambda_step*k1_p, rs);
    vec4 k2_p = dpdlambda(x + 0.5*lambda_step*k1_x, p + 0.5*lambda_step*k1_p, rs);
    vec4 k3_x = dxdlambda(x + 0.5*lambda_step*k2_x, p + 0.5*lambda_step*k2_p, rs);
    vec4 k3_p = dpdlambda(x + 0.5*lambda_step*k2_x, p + 0.5*lambda_step*k2_p, rs);
    vec4 k4_x = dxdlambda(x + lambda_step*k3_x, p + lambda_step*k3_p, rs);
    vec4 k4_p = dpdlambda(x + lambda_step*k3_x, p + lambda_step*k3_p, rs);
    x += lambda_step * (k1_x + 2.0*k2_x + 2.0*k3_x + k4_x) / 6.0;
    p += lambda_step * (k1_p + 2.0*k2_p + 2.0*k3_p + k4_p) / 6.0;
}

void trace_ray(vec4 x0, vec4 p0, float rs, out int hit, out vec4 final_position, out vec4 final_momentum) {
    vec4 x = x0;
    vec4 p = p0;

    float lambda_step = 0.01;

    for (int i = 0; i < 3000; i++) {
        runge_kutta_step(x, p, rs, lambda_step);

        if (x.y <= 1.1) {
            hit = 1;
            break;
        }
    }
    final_position = x;
    final_momentum = p;
}

// [
//  [1/sqrt(1 - rs/r), 0,                    0,            0                          ],
//  [0,                1/sqrt(1/(1 - rs/r)), 0,            0                          ],
//  [0,                0,                    1/sqrt(r**2), 0                          ],
//  [0,                0,                    0,             1/sqrt(r**2*sin(theta)**2)]
// ]
mat4 tetrad(vec4 x, float rs) {
    float r = x.y;
    float theta = x.z;
    float f = 1.0 - rs / r;

    mat4 e = mat4(0.0); // tetrad: local → coordinate
    e[0][0] = 1.0 / sqrt(f);
    e[1][1] = sqrt(f);
    e[2][2] = 1.0 / r;
    e[3][3] = 1.0 / (r * sin(theta));

    return e;
}

void minkowskiCartesianRayToSchwarzschildSphericalRay(
    vec4 minkowskiRayOrigin,
    vec4 minkowskiRayDirection,
    out vec4 x,
    out vec4 p,
    float rs
) {
    // --- Convert origin to spherical coordinates ---
    float x_t = minkowskiRayOrigin.x;

    vec3 pos = minkowskiRayOrigin.yzw;
    float r = max(length(pos), 1e-4);           // Avoid zero-length at origin
    float theta = acos(clamp(pos.z / r, -1.0, 1.0));  // Clamp to avoid NaNs
    float phi = atan(pos.y, pos.x);            // Standard spherical phi

    x = vec4(x_t, r, theta, phi);

    // --- Convert direction to spherical basis ---
    vec3 dir = normalize(minkowskiRayDirection.yzw);

    // Spherical basis vectors
    vec3 e_r     = pos / r;
    vec3 e_theta = vec3(cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta));
    vec3 e_phi   = vec3(-sin(phi), cos(phi), 0.0);

    // Project direction onto basis
    float v_r     = dot(dir, e_r);
    float v_theta = dot(dir, e_theta);
    float v_phi   = dot(dir, e_phi);

    // Compute p_t for null ray (H = 0) in local frame
    float p_t = sqrt(v_r*v_r + v_theta*v_theta + v_phi*v_phi);

    vec4 p_local = vec4(p_t, v_r, v_theta, v_phi);

    // --- Transform local inertial frame vector to coordinate basis ---
    mat4 e = tetrad(x, rs);
    p = e * p_local;

    // --- Convert to covariant momentum using Schwarzschild metric ---
    float f = max(1.0 - rs / x.y, 1e-4); // avoid division by zero
    float sin_theta = max(sin(theta), 1e-4);

    p.x = -p.x * f;                  // p_t = g_tt * p^t
    p.y = p.y / f;                   // p_r = g_rr * p^r
    p.z = p.z * (x.y * x.y);         // p_theta = g_theta_theta * p^theta
    p.w = p.w * (x.y * x.y * sin_theta * sin_theta); // p_phi = g_phi_phi * p^phi
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float rs = .9;

    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;

    vec4 cartesian_ray_origin = vec4(0., -6., 0., 0.);

    vec3 dir = normalize(vec3(0.9, uv - 0.5));
    vec4 ray_direction = vec4(1.0, dir);

    vec4 x0;
    vec4 p0;
    minkowskiCartesianRayToSchwarzschildSphericalRay(cartesian_ray_origin, ray_direction, x0, p0, rs);

    // float H =
    //     - p0.x*p0.x
    //     + p0.y*p0.y
    //     + p0.z*p0.z
    //     + p0.w*p0.w;

    // if (abs(H) > 1e-3) {
    //     discard;
    // }

    int hit = 0;
    vec4 final_position;
    vec4 final_momentum;
    trace_ray(x0, p0, rs, hit, final_position, final_momentum);

    // float px = final_momentum.y * sin(final_momentum.z) * cos(final_momentum.w);
    // float py = final_momentum.y * sin(final_momentum.z) * sin(final_momentum.w);
    // float pz = final_momentum.y * cos(final_momentum.z);

    // vec3 col = vec3(p0.yzw);
    // vec3 col = vec3(hit);
    vec3 col = vec3(final_momentum.yzw);

    // Output to screen
    fragColor = vec4(col, 1.0);
}