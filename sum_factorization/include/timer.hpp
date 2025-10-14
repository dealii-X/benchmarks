#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>


class Timer
{
public:
    void start()
    {
        m_StartTime = m_clock::now();
        m_bRunning  = true;
    }

    void stop()
    {
        m_EndTime  = m_clock::now();
        m_bRunning = false;
    }

    double elapsedNanoseconds()
    {
        auto endTime = m_bRunning ? m_clock::now() : m_EndTime;
        return std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - m_StartTime).count();
    }

    double elapsedSeconds()
    {
        return elapsedNanoseconds() / 1.0e9;
    }

private:
    using m_clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<m_clock> m_StartTime{};
    std::chrono::time_point<m_clock> m_EndTime{};
    bool m_bRunning = false;
};

#endif //TIMER_HPP