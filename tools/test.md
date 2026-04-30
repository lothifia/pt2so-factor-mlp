python tools/run_python_pt_reference.py

g++ -std=c++17 -O2 -Iruntime/include tools/run_dlopen_model.cpp -ldl -o build/run_dlopen_model

export PT2SO_MODEL_KEY_FILE=artifacts/factor_mlp/model.key
./build/run_dlopen_model --lib build/libmodel.so

python tools/compare_reference_outputs.py
