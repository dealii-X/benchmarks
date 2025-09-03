#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>


class Timer
{
public:
    void start()
    {
        m_StartTime = std::chrono::high_resolution_clock::now();
        m_bRunning  = true;
    }

    void stop()
    {
        m_EndTime  = std::chrono::high_resolution_clock::now();
        m_bRunning = false;
    }

    double elapsedNanoseconds()
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> endTime;

        if (m_bRunning)
        {
            endTime = std::chrono::high_resolution_clock::now();
        }
        else
        {
            endTime = m_EndTime;
        }

        return std::chrono::duration_cast<std::chrono::nanoseconds>(endTime -
                                                                    m_StartTime)
            .count();
    }

    double elapsedSeconds()
    {
        return elapsedNanoseconds() / 1.0e9;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_StartTime;
    std::chrono::time_point<std::chrono::system_clock> m_EndTime;
    bool m_bRunning = false;
};

#endif //TIMER_HPP