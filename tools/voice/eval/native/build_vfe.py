"""Builds the new front-end (vfe.h C API) as a host shared library from a worktree's CMake target.

python3 -I build_vfe.py <worktree> <out-dir>   -> <out-dir>/libod_vfe.{dylib,so}
"""
import os
import re
import subprocess
import sys


def main(root, out):
    cpp = os.path.join(root, 'app/src/main/cpp')
    cm = open(os.path.join(cpp, 'CMakeLists.txt')).read()
    i = cm.index('set(VOICE_FE_SOURCES')
    block = cm[i:cm.index('set_target_properties(voice_frontend', i)]
    var = {'CMAKE_SOURCE_DIR': cpp}
    for name, val in re.findall(r'set\((\w+)\s+(\$\{CMAKE_SOURCE_DIR\}[^\s)]*)\)', cm):
        var[name] = val.replace('${CMAKE_SOURCE_DIR}', cpp)

    def expand(s):
        return re.sub(r'\$\{(\w+)\}', lambda m: var.get(m.group(1), m.group(0)), s)

    def items(head):
        out_ = []
        for m in re.finditer(re.escape(head) + r'(.*?)\)', block, re.S):
            out_ += [expand(t) for t in m.group(1).split() if not t.startswith('#')]
        return out_

    srcs = items('set(VOICE_FE_SOURCES') + items('list(APPEND VOICE_FE_SOURCES')
    srcs = [s if os.path.isabs(s) else os.path.join(cpp, s) for s in srcs if s.endswith(('.c', '.cc', '.cpp'))]
    srcs = [s for s in srcs if 'jni' not in os.path.basename(s)]
    incs = [expand(t) for t in items('target_include_directories(voice_frontend PRIVATE')]
    defs = [t for t in items('target_compile_definitions(voice_frontend PRIVATE') if re.match(r'^[A-Z_]', t)]
    host = os.uname().sysname
    defs = [d for d in defs if d not in ('WEBRTC_LINUX', 'WEBRTC_ANDROID', 'PFFFT_SIMD_DISABLE')]
    defs += ['WEBRTC_MAC'] if host == 'Darwin' else ['WEBRTC_LINUX']
    if os.uname().machine in ('arm64', 'aarch64'):
        defs += ['WEBRTC_ARCH_ARM64', 'WEBRTC_HAS_NEON']
    else:
        defs += ['PFFFT_SIMD_DISABLE']
    os.makedirs(out, exist_ok=True)
    objdir = os.path.join(out, 'vfe_obj')
    os.makedirs(objdir, exist_ok=True)
    flags = ['-O2', '-fPIC', '-w', '-fno-exceptions', '-fno-rtti'] + ['-D' + d for d in sorted(set(defs))] \
        + ['-I' + p for p in incs]
    objs = []
    for s in srcs:
        o = os.path.join(objdir, re.sub(r'[^\w.]', '_', os.path.relpath(s, cpp)) + '.o')
        if not os.path.exists(o) or os.path.getmtime(o) < os.path.getmtime(s):
            cc = ['cc', '-std=c11'] if s.endswith('.c') else ['c++', '-std=c++17']
            subprocess.check_call(cc + flags + ['-c', s, '-o', o])
        objs.append(o)
    ext = 'dylib' if host == 'Darwin' else 'so'
    lib = os.path.join(out, f'libod_vfe.{ext}')
    subprocess.check_call(['c++', '-shared', '-o', lib] + objs + ['-lm'])
    print(lib)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
