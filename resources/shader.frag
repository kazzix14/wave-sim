#version 430

out vec4 color;

layout(std140) buffer MyBlock
{{
    vec4 values[{len}];
}};

uniform float width;
uniform float height;

void main()
{{
    uint x = uint((gl_FragCoord.x / width) * float({size}));
    uint y = uint((gl_FragCoord.y / height) * float({size}));

    uint i = (x+y*{size})/4;
    uint j = (x+y*{size})%4;

    vec4 value_vec = values[i];

    float value = 0;
    switch (j)
    {{
        case 0:
            value = value_vec.x;
            break;
        case 1:
            value = value_vec.y;
            break;
        case 2:
            value = value_vec.z;
            break;
        case 3:
            value = value_vec.w;
            break;
    }}

    color = vec4(value/20.0, value/5.0, value, 1.0);
}}
