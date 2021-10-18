#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <iostream>
#include <memory>
#include <fstream>
#include <time.h>
#include <math.h>
#include <random>
#include <string>
#include <GL/glut.h>
#include <Eigen/Dense>
#include "vec.h"

//using namespace std;
using std::cout;
using std::endl;

// PI constant
#define PI 3.141592653589793238f
// Small perturbation
#define EPS 0.000001f
// Number of particles, perfect square for initial placement
#define N_particles (30*30)
// Number of panels, multiple of 4 becase 4 walss
#define N_panels (4*12)


int nb_part_sqrt = sqrt(N_particles);
float dt = 0.0001f;
float dx_particles = 0.4f / int(nb_part_sqrt);
float dx_panels = 4.0f / float(N_panels);
float area = dx_particles * dx_particles;
float sigma2 = 0.001;
bool do_panel = true;
static constexpr size_t do_constraint = 1;
float regul = 1.0f;
float strength = 3.0f;

//Panel struct
class Panel {
public:
    Vecf2 start_pos;
    Vecf2 end_pos;
    Vecf2 mid_pos;
    Vecf2 normal;
    Vecf2 tangent;
    float gamma;

    // Default constructor
    Panel()
        : start_pos(Vecf2::zero()),
        end_pos(Vecf2::zero()),
        mid_pos(Vecf2::zero()),
        normal(Vecf2::zero()),
        tangent(Vecf2::zero()),
        gamma(0.0f) {

    };

    // Default destructor
    ~Panel() {};
};

class Particle {
public:
    Vecf2 pos;
    Vecf2 vel_particles;
    Vecf2 vel_panels;
    Vecf2 vel;   // vel_particles + vel_panels
    Vecf2 vel_old;
    float gamma;
    float vorticity;

    // Default constructor
    Particle()
        : pos(Vecf2::zero()),
        vel_particles(Vecf2::zero()),
        vel_panels(Vecf2::zero()),
        vel(Vecf2::zero()),
        vel_old(Vecf2::zero()),
        gamma(0.0f),
        vorticity(0.0f) {};

    // Default destructor
    ~Particle() {};
};

std::vector<Panel> panels(N_panels, Panel());
std::vector<Particle> particles(N_particles, Particle());
std::vector<std::vector<Vecf2>> panels_vel_influences(N_panels, std::vector<Vecf2>(N_panels, Vecf2::zero()));

// Eigen type, for least square solver
typedef Eigen::Matrix<float, N_panels + do_constraint, N_panels> MatXf;
typedef Eigen::Matrix<float, N_panels + do_constraint, 1> VecXf;
typedef Eigen::Matrix<float, N_panels, 1> VecXf_sol;
MatXf A = MatXf::Zero();
VecXf b = VecXf::Zero();
VecXf_sol x = VecXf_sol::Zero();


// GL Parameters
int win_id;
int win_x = 800;
int win_y = 800;
int start, vor_display;
int step;

static void time_evolution(void);
static void init_particles(void);
static void init_panels(void);
static void construct_A(void);


/*
----------------------------------------------------------------------
Utility
----------------------------------------------------------------------
*/

float smoothing_kernel(const float r_norm2) {
    return (1.0f - exp(-0.5 * r_norm2 / sigma2)) / (2.0f * PI);
}

/*
----------------------------------------------------------------------
Reset simulation
----------------------------------------------------------------------
*/

static void init(void)
{
    for (int i = 0; i < N_particles; i++) {
        //panels[i] = Panel();
        particles[i] = Particle();
    }

    init_particles();
    init_panels();
    construct_A();
    start = 0;
    step = 0;
}

/*
----------------------------------------------------------------------
Draw function
----------------------------------------------------------------------
*/

static void pre_display(void)
{
    glViewport(0, 0, win_x, win_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}


static void draw_particles(void)
{
    glPointSize(5.0);
    glBegin(GL_POINTS);

    for (int i = 0; i < N_particles; i++) {
        if (particles[i].gamma >= 0.0f) {
            glColor4f(particles[i].gamma * 1.0f, 0.0f, 0.0f, 0.5);
            glVertex2f(particles[i].pos.x, particles[i].pos.y);
        }
        else {
            glColor4f(0.0f, 0.0f, -particles[i].gamma * 1.0f, 0.5);
            glVertex2f(particles[i].pos.x, particles[i].pos.y);
        }
    }

    glEnd();
}

static void draw_Panel(void)
{
    // border
    glLineWidth(10.0);
    glBegin(GL_LINES);
    for (int i = 0; i < N_panels; i++) {
        glColor3f(0.5, 0.5, 0.5);
        // position i start
        glVertex2f(panels[i].start_pos.x, panels[i].start_pos.y);
        // position i end
        glVertex2f(panels[i].end_pos.x, panels[i].end_pos.y);
    }
    glEnd();   

    // mid_pos
    glPointSize(8.0);
    glBegin(GL_POINTS);
    for (int i = 0; i < N_panels; i++) {
        if (panels[i].gamma > 0.0f) {
            glColor3f(panels[i].gamma * 1.0f, 0.0f, 0.0f);
        }
        else {
            glColor3f(0.0f, 0.0f, -panels[i].gamma * 1.0f);
        }
        // glColor3f(0.0, 1.0, 0.0);
        glVertex2f(panels[i].mid_pos.x, panels[i].mid_pos.y);
    }
    glEnd();   
}

/*
----------------------------------------------------------------------
Panel method - no penetration
----------------------------------------------------------------------
 */

void init_panels() {
    assert(N_panels % 4 == 0);
    int N_panels_side = N_panels / 4;
    Vecf2 start = Vecf2::zero();
    Vecf2 end = Vecf2::zero();

    Vecf2 right = Vecf2(dx_panels, 0.0f);
    Vecf2 left = Vecf2(-dx_panels, 0.0f);
    Vecf2 up = Vecf2(0.0f, dx_panels);
    Vecf2 down = Vecf2(0.0f, -dx_panels);

    Vecf2 start_00 = Vecf2(0.0f, 0.0f);
    Vecf2 end_00 = start_00 + right;
    Vecf2 start_10 = Vecf2(1.0f, 0.0f);
    Vecf2 end_10 = start_10 + up;
    Vecf2 start_11 = Vecf2(1.0f, 1.0f);
    Vecf2 end_11 = start_11 + left;
    Vecf2 start_01 = Vecf2(0.0f, 1.0f);
    Vecf2 end_01 = start_01 + down;

    for (int i = 0; i < N_panels; i++) {
        if (i == 0 * N_panels_side) {
            start = start_00;
            end = end_00;
        }
        else if (i == 1 * N_panels_side) {
            start = start_10;
            end = end_10;
        }
        else if (i == 2 * N_panels_side) {
            start = start_11;
            end = end_11;
        }
        else if (i == 3 * N_panels_side) {
            start = start_01;
            end = end_01;
        }

        panels[i] = Panel();
        panels[i].start_pos = start;
        panels[i].end_pos = end;
        if (i >= 0 * N_panels_side && i < 1 * N_panels_side) {
            start += right;
            end += right;
        }
        else if (i >= 1 * N_panels_side && i < 2 * N_panels_side) {
            start += up;
            end += up;
        }
        else if (i >= 2 * N_panels_side && i < 3 * N_panels_side) {
            start += left;
            end += left;
        }
        else if (i >= 3 * N_panels_side && i < 4 * N_panels_side) {
            start += down;
            end += down;
        }
        panels[i].mid_pos = (panels[i].start_pos + panels[i].end_pos) / 2.0f;
        panels[i].tangent = panels[i].end_pos - panels[i].start_pos;
        panels[i].tangent /= panels[i].tangent.norm();
        panels[i].normal = Vecf2(-panels[i].tangent.y, panels[i].tangent.x);
        panels[i].normal /= panels[i].normal.norm();
    }
}



Vecf2 pos_to_local(const Vecf2 pos, const Vecf2 n_i, const Vecf2 normal_i) {
    float cos_theta = normal_i.y; // cos(theta)
    float sin_theta = normal_i.x; // sin(theta)
    // TODO: double check
    Vecf2 pos_trans = pos - n_i;
    Vecf2 pos_local = Vecf2(cos_theta * pos_trans.x - sin_theta * pos_trans.y,
        sin_theta * pos_trans.x + cos_theta * pos_trans.y);
    return pos_local;
}

Vecf2 vel_to_world(const Vecf2 vel_local, const Vecf2 n_i, const Vecf2 normal_i) {
    float cos_theta = normal_i.y; // cos(theta)
    float sin_theta = normal_i.x; // sin(theta)
    Vecf2 vel = Vecf2(cos_theta * vel_local.x + sin_theta * vel_local.y,
        -sin_theta * vel_local.x + cos_theta * vel_local.y);
    return vel;
}

Vecf2 compute_vel_from_particles(Vecf2 pos) {

    Vecf2 vel = { 0.0f, 0.0f };
    for (int i = 0; i < N_particles; i++) {
        Vecf2 r = pos - particles[i].pos;
        float r_norm2 = r.norm2();
        if (r_norm2 == 0.0f) {
            // do nothing. cant happen if pos = pos_i,
        }
        else {
            Vecf2 vel_temp = Vecf2(-r.y, r.x);
            vel_temp = vel_temp * particles[i].gamma * smoothing_kernel(r_norm2) / std::max(r_norm2, EPS);

            // notice that theres a "-"" in front of sum
            vel = vel - vel_temp;
        }
    }

    return vel;
}

Vecf2 compute_vel_from_panel(const int i, const Vecf2 pos) {

    Vecf2 node_i = panels[i].start_pos;
    Vecf2 normal_i = panels[i].normal;
    Vecf2 node_ip1 = panels[i].end_pos;

    Vecf2 local_pos = pos_to_local(pos, node_i, normal_i);
    Vecf2 local_node_i = pos_to_local(node_i, node_i, normal_i);
    Vecf2 local_node_ipi = pos_to_local(node_ip1, node_i, normal_i);

    float l_i = (local_node_ipi - local_node_i).norm();
    float r_i = (local_pos - local_node_i).norm();
    float r_ip1 = (local_pos - local_node_ipi).norm();
    float cos_beta = (r_i * r_i + r_ip1 * r_ip1 - l_i * l_i) / (2.0f * r_i * r_ip1);
    float beta = acos(cos_beta);
    if (isnan(beta)) {
        beta = PI;
    }

    Vecf2 local_vel = Vecf2(beta / (2.0f * PI), log((r_ip1 + EPS) / (r_i + EPS)) / (2.0f * PI));
    Vecf2 vel = vel_to_world(local_vel, node_i, normal_i);
    return vel;
}

void construct_u_ij(void) {
    for (int i = 0; i < N_panels; i++) {
        for (int j = 0; j < N_panels; j++) {
            panels_vel_influences[i][j] = compute_vel_from_panel(i, panels[j].mid_pos);
        }
    }
}

void construct_A(void) {
    // construct u_ij matrix
    construct_u_ij();

    // zero circulation constraint
    for (int i = 0; i < N_panels; i++) {
        A(0, i) = float(regul);
    }

    // negative vel normal constraint
    for (int j = 0; j < N_panels; j++) {
        for (int i = 0; i < N_panels; i++) {
            Vecf2 normal_j = panels[j].normal;
            A(j + do_constraint, i) = -panels_vel_influences[i][j].dot(normal_j);
        }
    }

}


void construct_b(void) {

    // zero circulation constraint
    b(0) = float(0.0f);

    // vel normal constraint
    for (int j = 0; j < N_panels; j++) {
        Vecf2 mid_j = panels[j].mid_pos;
        Vecf2 normal_j = panels[j].normal;
        Vecf2 vel_j = compute_vel_from_particles(mid_j);
        b(j + do_constraint) = vel_j.dot(normal_j);
    }
}

void ls_solve(void) {
    // linear least-square solve
    // x = A.householderQr().solve(b);
    x = A.colPivHouseholderQr().solve(b);
    // x = A.fullPivHouseholderQr().solve(b);

    for (int i = 0; i < N_panels; i++) {
        panels[i].gamma = x(i);
    }

}

Vecf2 compute_vel_from_panels(Vecf2 pos) {
    Vecf2 u = { 0.0f, 0.0f };
    for (int i = 0; i < N_panels; i++) {
        u = u + compute_vel_from_panel(i, pos) * panels[i].gamma;
    }
    return u;
}

void update_vel_particles() {
    for (int i = 0; i < N_particles; i++) {
        particles[i].vel_particles = compute_vel_from_particles(particles[i].pos);
    }
}

void update_vel_panels() {
    for (int i = 0; i < N_particles; i++) {
        particles[i].vel_panels = compute_vel_from_panels(particles[i].pos);
    }
}

void update_pos_particles() {
    for (int i = 0; i < N_particles; i++) {
        particles[i].vel = particles[i].vel_particles;
        if (do_panel) {
            particles[i].vel += particles[i].vel_panels;
        }
        // Adam-Bashforth 2nd order
        particles[i].pos += (particles[i].vel * 1.5f - particles[i].vel_old * 0.5f) * dt;
        particles[i].vel_old = particles[i].vel;
       
    }
}

/*
----------------------------------------------------------------------
GLUT callback routines
----------------------------------------------------------------------
*/
static void key_func(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 'x': init(); break;
    case 'q': case 27: exit(0); break;
    case 's': start = !start; break;
    }
}

static void reshape_func(int width, int height)
{
    glutSetWindow(win_id);
    glutReshapeWindow(width, height);

    win_x = width;
    win_y = height;
}

static void display_func(void)
{
    pre_display();

    draw_Panel();
    draw_particles();

    glutSwapBuffers();
}

static void idle_func(void)
{
    time_evolution();
    glutSetWindow(win_id);
    glutPostRedisplay();
    if (step % 10 == 0) {
        cout << "iteration: " << step << endl;
    }
}

static void open_glut_window(void) {
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(win_x, win_y);
    win_id = glutCreateWindow("Fluids");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();

    pre_display();

    glutKeyboardFunc(key_func);
    glutReshapeFunc(reshape_func);
    glutIdleFunc(idle_func);
    glutDisplayFunc(display_func);
}

/*
----------------------------------------------------------------------
Main
----------------------------------------------------------------------
*/
static void init_particles(void)
{
    int k = 0;
    for (int i = 0; i < nb_part_sqrt; i++) {
        for (int j = 0; j < nb_part_sqrt; j++) {
            // float pos_x = (0.5*nb_part_sqrt - float(i) - 0.5f)*dx;
            // float pos_y = (0.5*nb_part_sqrt - float(j) - 0.5f)*dx;
            // cache_particles_pos_particles_x[k] = (pos_x - pos_y)/sqrt(2.0f) + 0.65f;
            // cache_particles_pos_particles_y[k] = (pos_x + pos_y)/sqrt(2.0f) + 0.5f;
            // cache_particles_gamma[k] = strength * (cache_particles_pos_particles_y[k] - 0.5f);

            Vecf2 pos = Vecf2(float(i) - 0.5f, float(j) - 0.5f) * dx_particles;
            particles[k].pos = pos + Vecf2(0.5f);
            particles[k].gamma = strength * (particles[k].pos.y - particles[k].pos.x);
            k += 1;
            cout << "hello" << endl;
        }
    }
}


static void time_evolution(void)
{
    // Biot_Savart law - Particles vel
    update_vel_particles();

    // Panel Method - boundary induced vel 
    construct_b();
    ls_solve();
    update_vel_panels();

    // Advect particles
    update_pos_particles();
    step += 1;
}

int main(int argc, char** argv)
{
    win_x = 800;
    win_y = 800;

    glutInit(&argc, argv);
    open_glut_window();

    srand((int)time(0));

    init();

    glutMainLoop();

    exit(0);
}
