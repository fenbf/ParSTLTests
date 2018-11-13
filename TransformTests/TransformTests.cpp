
#include <algorithm>
#include <chrono>
#include <execution>
#include <iostream>
#include <vector>
#include <set>
#include <random>
#include "glm/vec4.hpp" // glm::vec4
#include "glm/geometric.hpp"
#include "glm/gtc/constants.hpp"

constexpr int RUN_TIMES = 5;

template <typename TFunc> void RunAndMeasure(const char* title, TFunc func)
{
	std::set<double> times;
	std::vector results(RUN_TIMES, func()); // invoke it for the first time...
	for (int i = 0; i < RUN_TIMES; ++i)
	{
		const auto start = std::chrono::steady_clock::now();
		const auto ret = func();
		const auto end = std::chrono::steady_clock::now();
		results[i] = ret;
		times.insert(std::chrono::duration<double, std::milli>(end - start).count());
	}

	std::cout << title << ":\t " << *times.begin() << "ms (max was " << *times.rbegin() << ") " << results[0] << '\n';
}

float GenRandomFloat(float lower, float upper) 
{
	// usage of thread local random engines allows running the generator in concurrent mode
	thread_local static std::default_random_engine rd;
	std::uniform_real_distribution<float> dist(lower, upper);
	return dist(rd);
}

void TestDoubleValue(const size_t vecSize)
{
	std::vector<double> vec(vecSize, 0.5);
	std::generate(vec.begin(), vec.end(), []() { return GenRandomFloat(-1.0f, 1.0f); });
	std::vector out(vec);

	std::cout << "v*2:\n";

	RunAndMeasure("std::transform   ", [&vec, &out] {
		std::transform(vec.begin(), vec.end(), out.begin(),
			[](double v) {
			return v * 2.0;
		}
		);
		return out[0];
	});

	RunAndMeasure("std::transform seq", [&vec, &out] {
		std::transform(std::execution::seq, vec.begin(), vec.end(), out.begin(),
			[](double v) {
			return v*2.0;
		}
		);
		return out[0];
	});

	RunAndMeasure("std::transform par", [&vec, &out] {
		std::transform(std::execution::par, vec.begin(), vec.end(), out.begin(),
			[](double v) {
			return v*2.0;
		}
		);
		return out[0];
	});

	RunAndMeasure("omp parallel for", [&vec, &out] {
		#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(vec.size()); ++i) //  'i': index variable in OpenMP 'for' statement must have signed integral type
			out[i] = vec[i]*2.0;

		return out[0];
	});

	//RunAndMeasure("using raw loop  ", [&vec, &out] {
	//	for (int i = 0; i < static_cast<int>(vec.size()); ++i) //  'i': index variable in OpenMP 'for' statement must have signed integral type
	//		out[i] = vec[i] * 2.0;

	//	return out.size();
	//});
}

void TestTrig(const size_t vecSize)
{
	std::vector<double> vec(vecSize, 0.5);
	std::generate(vec.begin(), vec.end(), []() { return GenRandomFloat(0.0f, 0.5f*glm::pi<float>()); });
	std::vector out(vec);

	std::cout << "sqrt(sin*cos):\n";

	RunAndMeasure("std::transform   ", [&vec, &out] {
		std::transform(vec.begin(), vec.end(), out.begin(),
			[](double v) {
			// we need to watch for the negative values! as sqrt will generate a domain error....
			return std::sqrt(std::sin(v)*std::cos(v));
		}
		);
		return out[0];
	});

	RunAndMeasure("std::transform seq", [&vec, &out] {
		std::transform(std::execution::seq, vec.begin(), vec.end(), out.begin(),
			[](double v) {
			return std::sqrt(std::sin(v)*std::cos(v));
		}
		);
		return out[0];
	});

	RunAndMeasure("std::transform par", [&vec, &out] {
		std::transform(std::execution::par, vec.begin(), vec.end(), out.begin(),
			[](double v) {
			return std::sqrt(std::sin(v)*std::cos(v));
		}
		);
		return out[0];
	});

	RunAndMeasure("omp parallel for", [&vec, &out] {
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(vec.size()); ++i) //  'i': index variable in OpenMP 'for' statement must have signed integral type
			out[i] = std::sqrt(::sin(vec[i])*std::cos(vec[i]));

		return out[0];
	});

	//RunAndMeasure("using raw loop  ", [&vec, &out] {
	//	for (int i = 0; i < static_cast<int>(vec.size()); ++i) //  'i': index variable in OpenMP 'for' statement must have signed integral type
	//		out[i] = std::sqrt(std::sin(vec[i])*std::cos(vec[i]));

	//	return out.size();
	//});
}

// implementation taken from
// https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel


float fresnel(const glm::vec4 &I, const glm::vec4 &N, const float ior)
{
	float cosi = std::clamp(glm::dot(I, N), -1.0f, 1.0f);
	float etai = 1, etat = ior;
	if (cosi > 0) { std::swap(etai, etat); }
	// Compute sini using Snell's law
	float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
	// Total internal reflection
	if (sint >= 1) 
		return 1.0f;

	float cost = sqrtf(std::max(0.f, 1 - sint * sint));
	cosi = fabsf(cosi);
	float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
	float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
	return (Rs * Rs + Rp * Rp) / 2.0f;

	// As a consequence of the conservation of energy, transmittance is given by:
	// kt = 1 - kr;
}

void TestFresnel(const size_t vecSize)
{
	std::vector<glm::vec4> vec(vecSize, { 0.0f, 1.0f, 0.0f, 1.0f });
	std::generate(vec.begin(), vec.end(), []() {
		return glm::vec4(GenRandomFloat(-1.0f, 1.0f), GenRandomFloat(-1.0f, 1.0f), GenRandomFloat(-1.0f, 1.0f), 1.0f);
	});
	std::vector<glm::vec4> vecNormals(vecSize, { 1.0f, 0.0f, 0.0f, 1.0f });
	std::generate(vec.begin(), vec.end(), []() {
		return glm::vec4(GenRandomFloat(-1.0f, 1.0f), GenRandomFloat(-1.0f, 1.0f), GenRandomFloat(-1.0f, 1.0f), 1.0f);
	});
	std::vector<float> vecFresnelTerms(vecSize);

	std::cout << "fresnel:\n";

	RunAndMeasure("std::transform   ", [&vec, &vecNormals, &vecFresnelTerms] {
		std::transform(vec.begin(), vec.end(), vecNormals.begin(), vecFresnelTerms.begin(),
			[](const glm::vec4& v, const glm::vec4& n) {
			return fresnel(v, n, 1.0f);
		}
		);
		return vecFresnelTerms[0];
	});

	RunAndMeasure("std::transform seq", [&vec, &vecNormals, &vecFresnelTerms] {
		std::transform(std::execution::seq, vec.begin(), vec.end(), vecNormals.begin(), vecFresnelTerms.begin(),
			[](const glm::vec4& v, const glm::vec4& n) {
			return fresnel(v, n, 1.0f);
		}
		);
		return vecFresnelTerms[0];
	});

	RunAndMeasure("std::transform par", [&vec, &vecNormals, &vecFresnelTerms] {
		std::transform(std::execution::par, vec.begin(), vec.end(), vecNormals.begin(), vecFresnelTerms.begin(),
			[](const glm::vec4& v, const glm::vec4& n) {
			return fresnel(v, n, 1.0f);
		}
		);
		return vecFresnelTerms[0];
	});

	RunAndMeasure("omp parallel for", [&vec, &vecNormals, &vecFresnelTerms] {
		#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(vec.size()); ++i) //  'i': index variable in OpenMP 'for' statement must have signed integral type
			vecFresnelTerms[i] = fresnel(vec[i], vecNormals[i], 1.0f);

		return vecFresnelTerms[0];
	});

	//RunAndMeasure("using raw loop  ", [&vec, &vecNormals, &vecFresnelTerms] {
	//	for (int i = 0; i < static_cast<int>(vec.size()); ++i) //  'i': index variable in OpenMP 'for' statement must have signed integral type
	//		vecFresnelTerms[i] = fresnel(vec[i], vecNormals[i], 1.0f);

	//	return vecFresnelTerms.size();
	//});
}

int main(int argc, char* argv[])
{
#ifdef _DEBUG
	const size_t vecSize = argc > 1 ? atoi(argv[1]) : 10000;
#else
	const size_t vecSize = argc > 1 ? atoi(argv[1]) : 6000000;
#endif
	std::cout << vecSize << '\n';
	std::cout << "Running each test " << RUN_TIMES << " times\n";
	
	int step = argc > 2 ? atoi(argv[2]) : 0;

	if (step == 0 || step == 1)
		TestDoubleValue(vecSize);
	
	if (step == 0 || step == 2)
		TestTrig(vecSize);
	
	if (step == 0 || step == 3)
		TestFresnel(vecSize);

	return 0;
}