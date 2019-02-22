// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef K4AWINDOWMANAGER_H
#define K4AWINDOWMANAGER_H

// Associated header
//

// System headers
//
#include <array>
#include <memory>
#include <stack>
#include <vector>

// Library headers
//
#include "k4aimgui_all.h"

// Project headers
//
#include "ik4adockcontrol.h"
#include "ik4avisualizationwindow.h"

namespace k4aviewer
{

class K4AWindowManager
{
public:
    static K4AWindowManager &Instance();

    void SetGLWindowSize(ImVec2 glWindowSize);
    void SetMenuBarHeight(float menuBarHeight);

    void AddWindow(std::unique_ptr<IK4AVisualizationWindow> &&window);
    void AddWindowGroup(std::vector<std::unique_ptr<IK4AVisualizationWindow>> &&windowGroup);
    void ClearFullscreenWindow();
    void ClearWindows();

    void PushDockControl(std::unique_ptr<IK4ADockControl> &&dockControl);
    void PopDockControl();

    void ShowAll();

    K4AWindowManager(const K4AWindowManager &) = delete;
    K4AWindowManager &operator=(const K4AWindowManager &) = delete;
    K4AWindowManager(const K4AWindowManager &&) = delete;
    K4AWindowManager &operator=(const K4AWindowManager &&) = delete;

private:
    K4AWindowManager() = default;
    struct WindowListEntry
    {
        WindowListEntry() : IsWindowGroup(true) {}

        WindowListEntry(std::unique_ptr<IK4AVisualizationWindow> &&window) :
            IsWindowGroup(false),
            Window(std::move(window))
        {
        }

        WindowListEntry(std::vector<std::unique_ptr<IK4AVisualizationWindow>> &&windowGroup) : IsWindowGroup(true)
        {
            for (auto &&windowEntry : windowGroup)
            {
                WindowGroup.emplace_back(std::move(windowEntry));
            }
        }

        bool IsWindowGroup;
        std::unique_ptr<IK4AVisualizationWindow> Window;
        std::vector<WindowListEntry> WindowGroup;
    };

    void ShowWindowArea(ImVec2 windowAreaPosition, ImVec2 windowAreaSize, WindowListEntry &windowList);
    void
    ShowWindow(ImVec2 windowAreaPosition, ImVec2 windowAreaSize, IK4AVisualizationWindow *window, bool isMaximized);

    void ShowDock();

    ImVec2 m_glWindowSize = { 0, 0 };
    float m_menuBarHeight = 0;
    float m_dockWidth = 0;

    IK4AVisualizationWindow *m_maximizedWindow = nullptr;

    WindowListEntry m_windows;

    std::stack<std::unique_ptr<IK4ADockControl>> m_dockControls;
};
} // namespace k4aviewer

#endif
