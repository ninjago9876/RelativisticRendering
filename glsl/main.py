import os
import time
import pygame
import moderngl
import array

WINDOW_SIZE = (800, 600)
SHADER_FILE = "screen_quad.frag"


def load_shader_file(filepath):
    with open(filepath, "r") as f:
        return f.read()


def create_shader_program(ctx, fragment_source):
    vertex_shader = """
        #version 330
        in vec2 in_vert;
        out vec2 uv;
        void main() {
            uv = in_vert * 0.5 + 0.5;
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
    """

    fragment_header = """
        #version 330
        uniform float iTime;
        uniform vec2 iResolution;
        uniform vec4 iMouse;
        uniform int iFrame;
        uniform float iTimeDelta;
        uniform float iFrameRate;
        uniform float iSampleRate;
        uniform vec4 iDate;
        uniform float iChannelTime[4];
        uniform vec3 iChannelResolution[4];
        uniform sampler2D iChannel0;
        uniform sampler2D iChannel1;
        uniform sampler2D iChannel2;
        uniform sampler2D iChannel3;
        out vec4 _frag_color;
    """

    full_fragment = fragment_header + "\n" + fragment_source + "\nvoid main() { mainImage(_frag_color, gl_FragCoord.xy); }"

    return ctx.program(vertex_shader=vertex_shader, fragment_shader=full_fragment)


def create_quad(ctx, prog):
    vertices = array.array('f', [
        -1.0, -1.0,
         1.0, -1.0,
         1.0,  1.0,
        -1.0,  1.0,
    ])
    indices = array.array('I', [
        0, 1, 2,
        2, 3, 0,
    ])
    vbo = ctx.buffer(vertices.tobytes())
    ibo = ctx.buffer(indices.tobytes())
    return ctx.vertex_array(prog, [(vbo, '2f', 'in_vert')], index_buffer=ibo)


def main():
    pygame.init()
    pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    ctx = moderngl.create_context()

    last_modified = os.path.getmtime(SHADER_FILE)
    fragment_source = load_shader_file(SHADER_FILE)
    program = create_shader_program(ctx, fragment_source)
    vao = create_quad(ctx, program)

    iFrame = 0
    clock = pygame.time.Clock()
    running = True
    width, height = WINDOW_SIZE

    while running:
        resized = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.w, event.h
                pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
                ctx.viewport = (0, 0, width, height)
                resized = True

        current_modified = os.path.getmtime(SHADER_FILE)
        if current_modified != last_modified:
            print("File changed, recompiling...")
            last_modified = current_modified
            try:
                fragment_source = load_shader_file(SHADER_FILE)
                program = create_shader_program(ctx, fragment_source)
                vao = create_quad(ctx, program)
                print("Recompiled successfully.")
            except Exception as e:
                print(f"Compile error: {e}")

        iTime = pygame.time.get_ticks() / 1000.0
        iTimeDelta = clock.tick() / 1000.0
        iFrameRate = 1.0 / iTimeDelta if iTimeDelta > 0 else 0
        mouse_buttons = pygame.mouse.get_pressed(3)
        mouse_pos = pygame.mouse.get_pos()
        iMouse = (
            float(mouse_pos[0]),
            float(height - mouse_pos[1]),
            float(mouse_buttons[0]),
            float(mouse_buttons[2]),
        )
        iDate = time.localtime()
        iDate = (float(iDate.tm_year), float(iDate.tm_mon), float(iDate.tm_mday),
                 float(iDate.tm_hour * 3600 + iDate.tm_min * 60 + iDate.tm_sec))

        # Set uniforms safely
        for name, value in {
            'iTime': iTime,
            'iResolution': (width, height),
            'iMouse': iMouse,
            'iFrame': iFrame,
            'iTimeDelta': iTimeDelta,
            'iFrameRate': iFrameRate,
            'iSampleRate': 44100.0,
            'iDate': iDate
        }.items():
            if name in program:
                program[name].value = value

        for i in range(4):
            if f'iChannelTime[{i}]' in program:
                program[f'iChannelTime[{i}]'].value = iTime
            if f'iChannelResolution[{i}]' in program:
                program[f'iChannelResolution[{i}]'].value = (1, 1, 1)

        ctx.screen.use()
        ctx.clear()
        vao.render()
        pygame.display.flip()
        iFrame += 1

    pygame.quit()


if __name__ == "__main__":
    main()