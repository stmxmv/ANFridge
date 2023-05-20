//
// Created by aojoie on 4/22/2023.
//

#define _GAMING_DESKTOP
#define WINAPI_FAMILY WINAPI_FAMILY_DESKTOP_APP

#define IMGUI_DEFINE_MATH_OPERATORS

#include <iostream>
#include <ojoie/Core/App.hpp>
#include <ojoie/Core/Window.hpp>
#include <ojoie/Core/Screen.hpp>
#include <ojoie/Core/Game.hpp>
#include <ojoie/Threads/Dispatch.hpp>
#include <ojoie/Camera/Camera.hpp>
#include <ojoie/Core/Configuration.hpp>

#include <ojoie/Render/TextureLoader.hpp>

#include <ojoie/Render/Mesh/MeshRenderer.hpp>

#include <ojoie/Core/Behavior.hpp>
#include <ojoie/Input/InputComponent.hpp>
#include <ojoie/Render/QualitySettings.hpp>

#include <ojoie/Core/Event.hpp>
#include <ojoie/Core/DragAndDrop.hpp>
#include <ojoie/IMGUI/IMGUI.hpp>
#include <ojoie/Utility/String.hpp>

#include <imgui_stdlib.h>

#define IMSPINNER_DEMO
#include "imspinner.h"
#include "ctpl.h"
#include <curl/curl.h>
#include <curl/easy.h>

#include <ANFridge/Inference.hpp>
#include <ANFridge/OCR.hpp>

#include <Windows.h>

/// this will make window use common control style instead of window classic style
#pragma comment(linker,"\"/manifestdependency:type='win32' \
name='Microsoft.Windows.Common-Controls' version='6.0.0.0' \
processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

using std::cout, std::endl;

#include <Windows.h>
#include <wrl\client.h>

using Microsoft::WRL::ComPtr;
using namespace AN;

AN::Window *gMainWindow;
constexpr int class_num = 28;

static const char *class_names[] = {
    "apple",
    "banana",
    "pear",
    "watermelon",
    "garlic",
    "lettuce",
    "onion",
    "pepper",
    "potato",
    "sweet_potato",
    "tomato",
    "cabbage",
    "carrot",
    "celery",
    "chinese_cabbage",
    "cucumber",
    "egg",
    "peach",
    "persimmon",
    "pitaya",
    "plum",
    "Pomegranate",
    "carambola",
    "guava",
    "kiwi",
    "mango",
    "muskmelon",
    "orange"
};

class LoggerGUI {
    ImGuiTextBuffer     Buf;
    ImGuiTextFilter     Filter;
    ImVector<int>       LineOffsets; // Index to lines offset. We maintain this with AddLog() calls.
    bool                AutoScroll;  // Keep scrolling if already at the bottom.

    int selected = 0;

    void contextMenu(int n, const char *log) {
        // <-- use last item id as popup id
        if (ImGui::BeginPopupContextItem()) {
            selected = n;
            if (ImGui::Button("Copy")) {
                ImGui::CloseCurrentPopup();
                ImGui::SetClipboardText(log);
            }
            ImGui::EndPopup();
        }
    }

public:
    LoggerGUI() {
        AutoScroll = true;
        clear();
    }

    void clear() {
        Buf.clear();
        LineOffsets.clear();
        LineOffsets.push_back(0);
    }

    void addLog(const char* fmt, ...) IM_FMTARGS(2) {
        int old_size = Buf.size();
        va_list args;
        va_start(args, fmt);
        Buf.appendfv(fmt, args);
        va_end(args);
        for (int new_size = Buf.size(); old_size < new_size; old_size++)
            if (Buf[old_size] == '\n')
                LineOffsets.push_back(old_size + 1);
    }

    void draw(const char* title, bool* p_open = nullptr) {
        if (!ImGui::Begin(title, p_open))
        {
            ImGui::End();
            return;
        }

        // Options menu
        if (ImGui::BeginPopup("Options"))
        {
            ImGui::Checkbox("Auto-scroll", &AutoScroll);
            ImGui::EndPopup();
        }

        // Main window
        if (ImGui::Button("Options"))
            ImGui::OpenPopup("Options");
        ImGui::SameLine();
        bool shouldClear = ImGui::Button("Clear");
        ImGui::SameLine();
        bool copy = ImGui::Button("Copy");
        ImGui::SameLine();
        Filter.Draw("Filter", -100.0f);

        ImGui::Separator();

        if (ImGui::BeginChild("scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar))
        {
            if (shouldClear)
                clear();
            if (copy)
                ImGui::LogToClipboard();

            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
            const char* buf = Buf.begin();
            const char* buf_end = Buf.end();
            int lineNumber      = 0;
            if (Filter.IsActive()) {
                // In this example we don't use the clipper when Filter is enabled.
                // This is because we don't have random access to the result of our filter.
                // A real application processing logs with ten of thousands of entries may want to store the result of
                // search/filter.. especially if the filtering function is not trivial (e.g. reg-exp).
                for (int line_no = 0; line_no < LineOffsets.Size; line_no++) {
                    const char *line_start = buf + LineOffsets[line_no];
                    char *line_end         = (char *) ((line_no + 1 < LineOffsets.Size) ? (buf + LineOffsets[line_no + 1] - 1) : buf_end);
                    if (Filter.PassFilter(line_start, line_end)) {
                        // ImGui::TextUnformatted(line_start, line_end);

                        char old    = *line_end;
                        line_end[0] = 0;
                        if (ImGui::Selectable(line_start, lineNumber == selected,
                                              ImGuiSelectableFlags_AllowDoubleClick, ImVec2(0, 0))) {
                            if (ImGui::IsMouseDoubleClicked(0)) {
                                // Double-click action, if desired
                            }
                            selected = lineNumber;
                        }

                        contextMenu(lineNumber, line_start);

                        line_end[0] = old;
                        ++lineNumber;
                    }
                }
            } else {
                // The simplest and easy way to display the entire buffer:
                //   ImGui::TextUnformatted(buf_begin, buf_end);
                // And it'll just work. TextUnformatted() has specialization for large blob of text and will fast-forward
                // to skip non-visible lines. Here we instead demonstrate using the clipper to only process lines that are
                // within the visible area.
                // If you have tens of thousands of items and their processing cost is non-negligible, coarse clipping them
                // on your side is recommended. Using ImGuiListClipper requires
                // - A) random access into your data
                // - B) items all being the  same height,
                // both of which we can handle since we have an array pointing to the beginning of each line of text.
                // When using the filter (in the block of code above) we don't have random access into the data to display
                // anymore, which is why we don't use the clipper. Storing or skimming through the search result would make
                // it possible (and would be recommended if you want to search through tens of thousands of entries).
                ImGuiListClipper clipper;
                clipper.Begin(LineOffsets.Size);

                while (clipper.Step()) {
                    for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++) {
                        const char *line_start = buf + LineOffsets[line_no];
                        char *line_end         = (char *) ((line_no + 1 < LineOffsets.Size) ? (buf + LineOffsets[line_no + 1] - 1) : buf_end);
                        //                        ImGui::TextUnformatted(line_start, line_end);

                        if (line_start[0] == 0) continue;

                        char old    = *line_end;
                        line_end[0] = 0;
                        if (ImGui::Selectable(line_start, lineNumber == selected,
                                              ImGuiSelectableFlags_AllowDoubleClick, ImVec2(0, 0))) {
                            if (ImGui::IsMouseDoubleClicked(0)) {
                                // Double-click action, if desired
                            }
                            selected = lineNumber;
                        }
                        contextMenu(lineNumber, line_start);
                        line_end[0] = old;
                        ++lineNumber;
                    }
                }
                clipper.End();
            }
            ImGui::PopStyleVar();

            // Keep up at the bottom of the scroll region if we were already at the bottom at the beginning of the frame.
            // Using a scrollbar or mouse-wheel will take away from the bottom edge.
            if (AutoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
                ImGui::SetScrollHereY(1.0f);
        }
        ImGui::EndChild();
        ImGui::End();
    }
};

class IMGUIDemo : public IMGUI {

    Texture2D *tex;
    float zoomLevel = 1.0f; // Initial zoom level
    bool isDragging = false;
    ImVec2 imagePosition{}; // Initial image position
    ImVec2 dragStartPosition{};

    ImVec2 lastPaintPosition{};

    bool bPaintStart = false;
    int brushRadius = 10;
    Vector3f brushColor{ 1.f, 0.f, 0.f }; // RGB
    cv::Mat paintingFrame;

    bool processing = false;
    bool showProcessSpinner = true;
    bool imageLoading = false;
    std::string textInput;

    std::string infoText;

    ctpl::thread_pool threadPool;

    std::unique_ptr<ANFridge::Inference> inference;
    std::unique_ptr<ANFridge::OCR> ocr;

    CURL *curl;
    std::vector<UInt8> imageBuffer; // for network

    LoggerGUI logger;

    bool paintMode = false;

    bool dragAndDropUpdating = false;

    bool dockFirstTime = true;

    static void OnToggleFullScreen(void *receiver, Message &message) {
        gMainWindow->setFullScreen(!gMainWindow->isFullScreen());
    }

    static size_t curl_write_data(void* contents, size_t size, size_t nmemb, IMGUIDemo *self) {
        size_t totalSize = size * nmemb;
        self->imageBuffer.insert(self->imageBuffer.end(), static_cast<char*>(contents), static_cast<char*>(contents) + totalSize);
        return totalSize;
    }

    static void ANLogCallback(const char *log, size_t size, void *userdata) {
        /// this can be call in any thread
        auto task = [log = std::string(log), userdata] {
            LoggerGUI *self = (LoggerGUI *)userdata;
            self->addLog("%s", log.c_str());
            fprintf(stderr, "%s", log.c_str());
            fflush(stderr);
        };

        if (GetCurrentThreadID() != Dispatch::GetThreadID(Dispatch::Game)) {
            Dispatch::async(Dispatch::Game, task);
        } else {
            task();
        }
    }


    void loadImageFromFile(const char *path) {
        imageLoading = true;

        threadPool.push([this, path = std::string(path)](int id) {
            auto result = TextureLoader::LoadTexture(path.c_str());

            Dispatch::async(Dispatch::Game, [this, result = std::move(result)] {
                if (result.isValid()) {
                    tex->resize(result.getWidth(), result.getHeight());
                    tex->setPixelData(result.getData());
                    tex->uploadToGPU(false);
                }
                imageLoading = false;
            });
        });
    }

    DECLARE_DERIVED_AN_CLASS(IMGUIDemo, IMGUI)

public:

    using IMGUI::IMGUI;

    static void InitializeClass() {
        GetClassStatic()->registerMessageCallback("OnToggleFullScreen", OnToggleFullScreen);
    }

    void dealloc() override {
        ANLogResetCallback();
        curl_easy_cleanup(curl);
        Super::dealloc();
    }

    virtual bool init() override {
        if (!Super::init()) return false;
        auto result = TextureLoader::LoadTexture("C:\\Users\\aojoie\\Desktop\\detect\\apples-1296x728-header.jpg");

        tex = NewObject<Texture2D>();

        TextureDescriptor descriptor{};
        descriptor.width = result.getWidth();
        descriptor.height = result.getHeight();
        descriptor.mipmapLevel = 1;
        descriptor.pixelFormat = result.getPixelFormat();

        SamplerDescriptor samplerDescriptor = Texture::DefaultSamplerDescriptor();
        samplerDescriptor.magFilter = AN::kSamplerMinMagFilterNearest;

        if (!tex->init(descriptor, samplerDescriptor)) return false;
        tex->setPixelData(result.getData());
        tex->setReadable(true);
        tex->uploadToGPU(false);


        /// init inference
        bool runOnGPU = false;
        inference = std::make_unique<ANFridge::Inference>("./models/fridge.onnx", class_num, cv::Size(640, 640), runOnGPU);

        int gpu[] = { 0 };
        ocr = std::make_unique<ANFridge::OCR>("./models/det/det.onnx",
                                              "./models/cls",
                                              "./models/rec/rec.onnx",
                                              "./models/ppocr_keys_v1.txt");

        threadPool.resize(1);

        curl = curl_easy_init();

        ANLogSetCallback(ANLogCallback, &logger);

        return true;
    }

    void imageZoom(float val) {
        zoomLevel += val; // Increase the zoom level
        clampZoom();
    }

    void clampZoom() {
        if (zoomLevel < 0.1f) {
            zoomLevel = 0.1f; // Limit the minimum zoom level
        }
    }

    cv::Mat getTextureCVMat() {
        void *rawData = tex->getPixelData();
        ANAssert(rawData != nullptr);

        switch (tex->getPixelFormat()) {
            case kPixelFormatRGBA8Unorm_sRGB:
            case kPixelFormatRGBA8Unorm:
                break;
            default:
                AN_LOG(Error, "unsupport image format");
                return {};
        }

        cv::Mat rawFrame(tex->getDataHeight(), tex->getDataWidth(), CV_8UC4, rawData);

        cv::Mat frame;
        cv::cvtColor(rawFrame, frame, cv::COLOR_RGBA2BGR);
        return frame;
    }

    void ToggleButton(const char* str_id, bool* v) {
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        float height = ImGui::GetFrameHeight();
        float width = height * 1.55f;
        float radius = height * 0.50f;

        ImGui::InvisibleButton(str_id, ImVec2(width, height));
        if (ImGui::IsItemClicked())
            *v = !*v;

        float t = *v ? 1.0f : 0.0f;

        ImGuiContext& g = *GImGui;
        float ANIM_SPEED = 0.08f;
        if (g.LastActiveId == g.CurrentWindow->GetID(str_id))// && g.LastActiveIdTimer < ANIM_SPEED)
        {
            float t_anim = ImSaturate(g.LastActiveIdTimer / ANIM_SPEED);
            t = *v ? (t_anim) : (1.0f - t_anim);
        }

        ImU32 col_bg;
        if (ImGui::IsItemHovered())
            col_bg = ImGui::GetColorU32(ImLerp(ImVec4(0.78f, 0.78f, 0.78f, 1.0f), ImVec4(0.64f, 0.83f, 0.34f, 1.0f), t));
        else
            col_bg = ImGui::GetColorU32(ImLerp(ImVec4(0.85f, 0.85f, 0.85f, 1.0f), ImVec4(0.56f, 0.83f, 0.26f, 1.0f), t));

        draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5f);
        draw_list->AddCircleFilled(ImVec2(p.x + radius + t * (width - radius * 2.0f), p.y + radius), radius - 1.5f, IM_COL32(255, 255, 255, 255));
    }

    void HelpMarker(const char* desc) {
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
        {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(desc);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    virtual void onGUI() override {
        ImGuiIO& io = ImGui::GetIO();

        {
            ImGuiViewport *viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(viewport->WorkPos);
            ImGui::SetNextWindowSize(viewport->WorkSize);
            ImGui::SetNextWindowViewport(viewport->ID);

            ImGuiWindowFlags host_window_flags = 0;
            host_window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking;
            host_window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
//            host_window_flags |= ImGuiWindowFlags_MenuBar;
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

            ImGui::Begin("MainDockSpaceViewport", nullptr, host_window_flags);
            ImGui::PopStyleVar(3);

            ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");
            ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f));

            ImGuiDockNodeFlags dockspace_flags = 0;

            if (dockFirstTime) {
                dockFirstTime = false;

                ImGui::DockBuilderRemoveNode(dockspace_id);
                ImGui::DockBuilderAddNode(dockspace_id, dockspace_flags | ImGuiDockNodeFlags_DockSpace);
                ImGui::DockBuilderSetNodeSize(dockspace_id, viewport->Size);

                ImGuiID dock_id_up;
                ImGuiID dock_id_down = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Down, 0.2f, nullptr, &dock_id_up);

                ImGui::DockBuilderDockWindow("Logger", dock_id_down);

                ImGuiID dock_id_right;
                ImGuiID dock_id_left = ImGui::DockBuilderSplitNode(dock_id_up, ImGuiDir_Left, 0.3f, nullptr, &dock_id_right);

                ImGui::DockBuilderDockWindow("Info", dock_id_left);
                ImGui::DockBuilderDockWindow("Main Window", dock_id_right);



                ImGui::DockBuilderFinish(dockspace_id);
            }
            /*
            if (ImGui::BeginMenuBar())
            {
                if (ImGui::BeginMenu("App"))
                {
                    // Disabling fullscreen would allow the window to be moved to the front of other windows,
                    // which we can't undo at the moment without finer window depth/z control.

                    if (ImGui::MenuItem("Open")) {

                    }

                    if (ImGui::MenuItem("Close")) {
                        App->terminate();
                    }

                    ImGui::EndMenu();
                }
                HelpMarker(
                        "When docking is enabled, you can ALWAYS dock MOST window into another! Try it now!" "\n"
                        "- Drag from window title bar or their tab to dock/undock." "\n"
                        "- Drag from window menu button (upper-left button) to undock an entire node (all windows)." "\n"
                        "- Hold SHIFT to disable docking (if io.ConfigDockingWithShift == false, default)" "\n"
                        "- Hold SHIFT to enable docking (if io.ConfigDockingWithShift == true)" "\n"
                        "This demo app has nothing to do with enabling docking!" "\n\n"
                        "This demo app only demonstrate the use of ImGui::DockSpace() which allows you to manually create a docking node _within_ another window." "\n\n"
                        "Read comments in ShowExampleAppDockSpace() for more details.");

                ImGui::EndMenuBar();
            } */

        }

        static bool show_demo_window = false;
        static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
        {
            ImGui::Begin("Info");

            ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f * GetGame().deltaTime, 1.f / GetGame().deltaTime);

            if (!infoText.empty()) {
                ImGui::Separator();
                ImGui::Text("%s", infoText.c_str());
            }

            ImGui::End();
        }

        logger.draw("Logger");

        {
            ImGui::Begin("Main Window");


            ImGui::SameLine();
            if (ImGui::Button("Zoom")) {
                ImGui::OpenPopup("Zoom");
            }
            if (ImGui::BeginPopup("Zoom")) {
                ImGui::DragFloat("Zoom", &zoomLevel, zoomLevel * 0.1f, 0.1f);
                clampZoom();
                ImGui::EndPopup();
            }

            ImGui::SameLine();
            if (ImGui::Button("Reset")) {
                zoomLevel = 1.f;
                imagePosition = {};
            }

            ImGui::SameLine();

            if (ImGui::Button("OpenFile") && !imageLoading && !processing) {
                OpenPanel *openPanel = OpenPanel::Alloc();

                if (!openPanel->init()) {
                    AN_LOG(Error, "init openPannel fail");
                } else {
                    openPanel->setAllowOtherTypes(true);
                    openPanel->addAllowContentExtension("Image", "jpg");
                    openPanel->addAllowContentExtension("Image", "png");
                    openPanel->addAllowContentExtension("Image", "jpeg");
                    openPanel->addAllowContentExtension("Image", "jiff");

                    openPanel->beginSheetModal(gMainWindow, [this](ModalResponse res, const char *path) {
                        if (res == AN::kModalResponseOk && path) {
                            AN_LOG(Log, "Load image %s", path);

                            loadImageFromFile(path);
                        }
                    });
                }
                openPanel->release();
            }

            ImGui::SameLine();

            if (ImGui::Button("Open From URL") && !processing && !imageLoading) {
                ImGui::OpenPopup("Open URL");
            }

            // Always center this window when appearing
            ImVec2 center = ImGui::GetMainViewport()->GetCenter();
            ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

            if (ImGui::BeginPopupModal("Open URL", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::Text("Enter URL:");
                ImGui::InputText("##input", &textInput, ImGuiInputTextFlags_EnterReturnsTrue);

                ImGui::SameLine();

                if (ImGui::Button("Paste")) {
                    const char *s = ImGui::GetClipboardText();
                    if (s) {
                        textInput = s;
                    }
                }

                if (ImGui::Button("OK", ImVec2(240, 0))) {
                    ImGui::CloseCurrentPopup();

                    if (!textInput.empty()) {

                        imageLoading = true;

                        threadPool.push([this](int id) {
                            curl_easy_setopt(curl, CURLOPT_URL, textInput.c_str());

                            // Set the write callback function and the buffer
                            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_data);
                            curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);
                            CURLcode curlResult = curl_easy_perform(curl);

                            if (curlResult != CURLE_OK) {
                                std::cerr << "Failed to download image. Error: " << curl_easy_strerror(curlResult) << std::endl;

                                Dispatch::async(Dispatch::Game, [this] {
                                    imageLoading = false;
                                });

                            } else {
                                std::cout << "Image downloaded successfully. Size: " << imageBuffer.size() << " bytes." << std::endl;

                                auto result = TextureLoader::LoadTextureFromMemory(imageBuffer.data(), imageBuffer.size());

                                std::vector<UInt8>().swap(imageBuffer); // free memory

                                Dispatch::async(Dispatch::Game, [this, result = std::move(result)] {
                                    if (result.isValid()) {
                                        tex->resize(result.getWidth(), result.getHeight());
                                        tex->setPixelData(result.getData());
                                        tex->uploadToGPU(false);
                                    }
                                    imageLoading = false;
                                });
                            }


                        });
                    }
                }

                ImGui::SetItemDefaultFocus();
                ImGui::SameLine();
                if (ImGui::Button("Cancel", ImVec2(240, 0))) { ImGui::CloseCurrentPopup(); }

                ImGui::EndPopup();
            }


            ImGui::SameLine();

            if (ImGui::Button("Save As")  && !processing && !imageLoading) {
                SavePanel *savePanel = SavePanel::Alloc();

                if (!savePanel->init()) {
                    AN_LOG(Error, "init openPanel fail");
                } else {
                    savePanel->setAllowOtherTypes(true);
                    savePanel->addAllowContentExtension("PNG", "png");
                    savePanel->addAllowContentExtension("JPG", "jpg");
                    savePanel->addAllowContentExtension("BMP", "bmp");

                    savePanel->setDefaultExtension("png");
                    savePanel->setFileName("Output.png");

                    savePanel->beginSheetModal(gMainWindow, [this, &savePanel](ModalResponse res, const char *path) {
                        if (res == AN::kModalResponseOk && path) {
                            AN_LOG(Info, "Save image at %s", path);

                            imageLoading = true;

                            std::string savePath(path);

                            ImageEncodeType type = kImageEncodeTypePNG;
                            if (strcmp(savePanel->getExtension(), "png") == 0) {
                                type = kImageEncodeTypePNG;
                            } else if (strcmp(savePanel->getExtension(), "jpg") == 0) {
                                type = AN::kImageEncodeTypeJPG;
                            } else if (strcmp(savePanel->getExtension(), "bmp") == 0) {
                                type = AN::kImageEncodeTypeBMP;
                            }

                            std::string_view ext = GetPathNameExtension(path);

                            if (ext.empty()) {
                                savePath.append(".");
                                savePath.append(savePanel->getExtension());
                            }

                            void *rawImageData = tex->getPixelData();

                            TextureLoader::EncodeTexture(savePath.c_str(), type,
                                                         rawImageData,
                                                         tex->getDataWidth(), tex->getDataHeight(),
                                                         AN::kPixelFormatRGBA8Unorm_sRGB,
                                                         90);

                            imageLoading = false;
                        }
                    });
                }
                savePanel->release();
            }

            ImGui::SameLine();

            if (ImGui::Button("Detect Fruit") && !processing && !imageLoading) {

                processing = true;
                cv::Mat frame = getTextureCVMat();
                threadPool.push([this, frame](int id) {


                    if (frame.empty()) {
                        Dispatch::async(Dispatch::Game, [this] {
                            processing = false;
                        });
                        return;
                    }

                    std::vector<ANFridge::Detection> output = inference->inference(frame);
                    int detections = output.size();

                    std::string info;
                    info.append(std::format("Number of detections: {}", detections));
                    for (int i = 0; i < detections; ++i) {
                        ANFridge::Detection detection = output[i];

                        cv::Rect box     = detection.box;
                        cv::Scalar color = detection.color;

                        // Detection box
                        cv::rectangle(frame, box, color, 2 * frame.rows / 728);

                        // Detection box text
                        std::string classString = std::to_string(detection.class_id) + ' ' + class_names[detection.class_id]
                                                  + std::to_string(detection.confidence).substr(0, 4);

                        classString = std::format("{} {} {:.4f}", detection.class_id, class_names[detection.class_id], detection.confidence);

                        info.push_back('\n');
                        info.append(classString);

                        cv::Size textSize       = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

                        cv::rectangle(frame, textBox, color, cv::FILLED);
                        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
                    }

                    cv::Mat showFrame;
                    cv::cvtColor(frame, showFrame, cv::COLOR_BGR2RGBA);

                    tex->setPixelData(showFrame.data);
                    tex->uploadToGPU(false);

                    Dispatch::async(Dispatch::Game, [this, info = std::move(info)] {
                        processing = false;
                        infoText = info;
                    });
                });


            }

            ImGui::SameLine();

            if (ImGui::Button("Detect Text") && !processing && !imageLoading) {
                processing = true;

                cv::Mat frame = getTextureCVMat();
                threadPool.push([this, frame](int id) {

                    if (frame.empty()) {
                        Dispatch::async(Dispatch::Game, [this] {
                            processing = false;
                        });
                        return;
                    }

                    std::vector<cv::Mat> images{ frame };
                    auto results = ocr->ocr(images);

                    std::string info;
                    info.append("OCR Result");
                    for (const auto &imageResult : results) {
                        for (const auto &text : imageResult) {
                            info.push_back('\n');
                            info.append(text.text);
                        }
                    }

                    Dispatch::async(Dispatch::Game, [this, info = std::move(info)] {
                        infoText = info;
                        processing = false;
                    });
                });
            }

            ImGui::SameLine();
            ImGui::Text("Paint Mode");
            ImGui::SameLine();
            ToggleButton("Paint Mode Toggle", &paintMode);

            ImGui::SameLine();

            if (ImGui::Button("Brush Options")) {
                ImGui::OpenPopup("Brush Options Popup");
            }

            if (ImGui::BeginPopup("Brush Options Popup")) {
                ImGui::SliderInt("Brush Radius", &brushRadius, 1, 100);
                ImGui::ColorEdit3("Brush Color", (float *)&brushColor);
                ImGui::EndPopup();
            }

            ImVec2 size = ImGui::GetContentRegionAvail();

            float imageRatio = (float)tex->getDataWidth() / (float)tex->getDataHeight();
            float imageHeight = size.y;
            float imageWidth = imageRatio * imageHeight;
            if (imageWidth > size.x) {
                imageWidth = size.x;
                imageHeight = imageWidth / imageRatio;
            }

            ImVec2 imageSize{ imageWidth * zoomLevel, imageHeight * zoomLevel };

            ImGui::BeginChild("Image", {}, false, ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoScrollbar);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
//            ImGui::PushClipRect(ImGui::GetCursorScreenPos(),
//                                ImVec2(ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x,
//                                       ImGui::GetCursorScreenPos().y + ImGui::GetContentRegionAvail().y),
//                                true);

            ImVec2 imageCursorPos = ImGui::GetCursorPos();
            ImVec2 imageCenterPos = imageCursorPos + size * 0.5f;
            ImVec2 imageBlockBegin = imageCursorPos + (size - imageSize) * 0.5f;
            ImGui::SetCursorPos(imageBlockBegin + imagePosition);

            ImGui::Image((void *)tex->getTextureID(), imageSize);

            if (Event::Current().getType() == AN::kDragExited) {
                dragAndDropUpdating = false;
            }

            if (dragAndDropUpdating) {
                ImDrawList* drawList = ImGui::GetWindowDrawList();
                ImVec2 startPos = imageBlockBegin + imagePosition + ImGui::GetWindowPos(); // Starting position of the rectangle
                ImVec2 endPos = startPos + imageSize; // Ending position of the rectangle
                ImU32 borderColor = IM_COL32(55, 142, 240, 255); // Border color (red in this example)
                float borderWidth = 2.0f; // Border width in pixels

                drawList->AddRect(startPos, endPos, borderColor, 0.0f, ImDrawCornerFlags_All, borderWidth);
            }

            // Handle mouse input for dragging the image
            if (ImGui::IsItemHovered()) {
                /// drag and drop
                if (Event::Current().getType() == AN::kDragUpdated) {
                    if (GetDragAndDrop().getPaths().size() == 1) {
                        GetDragAndDrop().setVisualMode(AN::kDragOperationCopy);

                        dragAndDropUpdating = true;
                    }

                } else if (Event::Current().getType() == kDragPerform) {
                    if (GetDragAndDrop().getPaths().size() == 1) {
                        loadImageFromFile(GetDragAndDrop().getPaths()[0].c_str());
                    }

                    dragAndDropUpdating = false;
                }

                if (ImGui::IsMouseDown(0) && !isDragging) {
                    isDragging = true;
                    dragStartPosition = ImGui::GetMousePos() - imagePosition;
                }
                float scroll = ImGui::GetIO().MouseWheel;

                if (std::abs(scroll) > 0.01f) {
                    float zoomSpeed = 0.1f;  // Adjust the zoom speed as needed
                    float zoomValue = scroll * zoomSpeed;

                    float oldZoomLevel = zoomLevel;
                    imageZoom(zoomValue);

//                    imageSize *= zoomLevel;

                    ImVec2 mousePosScreen = ImGui::GetMousePos();
                    ImVec2 windowPosScreen = ImGui::GetWindowPos();
                    ImVec2 mousePosWindow = mousePosScreen - windowPosScreen;
                    ImVec2 cursorPosRel = mousePosWindow - (imageCenterPos + imagePosition);
                    ImVec2 zoomPivot = ImVec2(cursorPosRel.x / size.x, cursorPosRel.y / size.y);
                    imagePosition -= cursorPosRel / oldZoomLevel * zoomLevel - cursorPosRel;

                }
            } else {
                dragAndDropUpdating = false;
            }

            if (isDragging) {
                if (paintMode) {
                    if (ImGui::IsItemHovered()) {
                        ImVec2 mousePosScreen  = ImGui::GetMousePos();
                        ImVec2 windowPosScreen = ImGui::GetWindowPos();
                        ImVec2 mousePosWindow  = mousePosScreen - windowPosScreen;
                        ImVec2 cursorPosRel    = mousePosWindow - (imageBlockBegin + imagePosition);

                        ImVec2 imagePixelSize{(float) tex->getDataWidth(), (float) tex->getDataHeight()};

                        ImVec2 lastPixelPos = lastPaintPosition;
                        ImVec2 pixelPos = cursorPosRel / imageSize * imagePixelSize;

                        if (!bPaintStart) {
                            bPaintStart = true;
                            processing = true;
                            showProcessSpinner = false;
                            lastPixelPos = pixelPos;

                            paintingFrame = getTextureCVMat();
                        }

                        lastPaintPosition = pixelPos;

                        threadPool.push([this, pixelPos, lastPixelPos](int id) {

                            if (paintingFrame.empty()) {
                                Dispatch::async(Dispatch::Game, [this] {
                                    processing = false;
                                });
                                return;
                            }

                            int step = 0; // lerp to step 100
                            int maxStep = 100;
                            for (; step <= maxStep; ++step) {
                                float t = (float)step / (float)maxStep;

                                Vector2f pos{ std::lerp(pixelPos.x, lastPixelPos.x, t),
                                             std::lerp(pixelPos.y, lastPixelPos.y, t) };

                                cv::circle(paintingFrame, {(int) pos.x, (int) pos.y}, brushRadius,
                                           cv::Scalar(brushColor.b * 255.0, brushColor.g * 255.0, brushColor.r * 255.0), cv::FILLED);
                            }


                            cv::Mat showFrame;
                            cv::cvtColor(paintingFrame, showFrame, cv::COLOR_BGR2RGBA);

                            Dispatch::async(Dispatch::Game, [this, showFrame] {
                                tex->setPixelData(showFrame.data);
                                tex->uploadToGPU(false);
                            });
                        });
                    }

                    if (ImGui::IsMouseReleased(0)) {
                        isDragging = false;
                        bPaintStart = false;

                        /// sync to work thread
                        threadPool.push([this](int id) {
                            Dispatch::async(Dispatch::Game, [this] {
                                if (!bPaintStart) {
                                    processing = false;
                                    showProcessSpinner = true;
                                    paintingFrame.release();
                                }
                            });
                        });
                    }

                } else {

                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);

                    if (ImGui::IsMouseReleased(0)) {
                        isDragging = false;
                    } else {
                        imagePosition = ImGui::GetMousePos() - dragStartPosition;
                    }
                }
            }

            if ((processing || imageLoading) && showProcessSpinner) {
                float offsetX = (size.x - 32.f * 2.f) / 2.f;
                float offsetY = (size.y - 32.f * 2.f) / 2.f;
                ImGui::SetCursorPos({ imageCursorPos.x + offsetX , imageCursorPos.y + offsetY });
                ImSpinner::SpinnerTwinAng360("label", 32, 22, 4, ImSpinner::white, { 0.0700f, 0.659f, 1.00f, 1.f});
            }

//            ImGui::PopClipRect();
            ImGui::PopStyleVar();

            ImGui::EndChild();

            ImGui::End();

//            ImGui::Begin("Test Window");
//            static bool selectableTextEnabled = false;
//            std::string text = "Read-only Text";
//
//            // Render the selectable text
//            if (ImGui::Selectable(text.c_str(), &selectableTextEnabled, ImGuiSelectableFlags_AllowDoubleClick, ImVec2(0, 0)))
//            {
//                if (ImGui::IsMouseDoubleClicked(0))
//                {
//                    // Double-click action, if desired
//                }
//            }
////            ImSpinner::demoSpinners();
//            ImGui::End();

            ImGui::End();// MainDockSpace
        }
    }
};

IMPLEMENT_AN_CLASS_HAS_INIT(IMGUIDemo)
LOAD_AN_CLASS(IMGUIDemo)

IMGUIDemo::~IMGUIDemo() {}


class AppDelegate : public AN::ApplicationDelegate {

    AN::RefCountedPtr<AN::Window> mainWindow;
    AN::RefCountedPtr<AN::Menu> mainMenu;
    AN::RefCountedPtr<AN::Menu> appMenu;
    AN::Size size;
public:

    virtual bool applicationShouldTerminateAfterLastWindowClosed(AN::Application *application) override {
        return true;
    }

    virtual void applicationDidFinishLaunching(AN::Application *application) override {

        //        MessageBoxW(nullptr, L"some text", L"caption", MB_OK);

        class AboutMenuDelegate : public MenuInterface {
        public:
            virtual void execute(const MenuItem &menuItem) override {
//                MessageBoxW(nullptr, L"about content", L"About", MB_OK);
                App->showAboutWindow();
            }
        };

        class CloseMenuDelegate : public MenuInterface {
        public:
            virtual void execute(const MenuItem &menuItem) override {
                App->terminate();
            }
        };
        AboutMenuDelegate *about = new AboutMenuDelegate();
        CloseMenuDelegate *close = new CloseMenuDelegate();

        appMenu = ref_transfer(Menu::Alloc());
        ANAssert(appMenu->init(AN::kMenuPopup));

        appMenu->addMenuItem("About #%a", "About", about);
        appMenu->addMenuItem("Quit #%q", "Quit", close);
        about->release();
        close->release();

        mainMenu = ref_transfer(Menu::Alloc());
        ANAssert(mainMenu->init(AN::kMenuTopLevel));
        mainMenu->addSubMenu("App", appMenu.get());


        size = AN::GetScreen().getSize();

        /// set the render target size as screen size no matter what window size is
        /// otherwise it will automatically set to window size, and it will not automatically change when you resize window,
        /// it may require recreate the whole framebuffer which causes large overhead instead it will only recreate
        /// the swapchain image
        GetQualitySettings().setTargetResolution(size);

        mainWindow = AN::ref_transfer(AN::Window::Alloc());
        //        AN::Size screenSize = AN::GetDefaultScreenSize();
        mainWindow->init({0, 0, size.width * 4 / 5, size.height * 4 / 5 });
        mainWindow->setMenu(mainMenu.get());
        mainWindow->setTitle("图片测试");
        mainWindow->center();
        mainWindow->makeKeyAndOrderFront();

        gMainWindow = mainWindow.get();

        //        AN::Cursor::setState(AN::kCursorStateDisabled);
        //        mainWindow->setCursorState(AN::kCursorStateDisabled);
        ANAssert(AN::App->getMainWindow() == mainWindow.get());
    }

    virtual void applicationWillTerminate(AN::Application *application) override {
        /// release resources here
        mainWindow = nullptr;

    }

    InputActionMap *createInputActionMap() {
        InputActionMap *actionMap = NewObject<InputActionMap>();
        actionMap->init("MainInput");
        InputAction &action = actionMap->addAction("Move", kInputControlVector2);
        action.addVector2CompositeBinding(kInputKey_W, kInputKey_S, kInputKey_A, kInputKey_D);

        /// when add another action, above action may become invalid pointer
        InputAction &lookAction = actionMap->addAction("Look", kInputControlVector2);
        lookAction.addBinding(kPointerDelta);

        InputAction &terminateAction = actionMap->addAction("Terminate", AN::kInputControlButton);
        terminateAction.addBinding(kInputKeyEsc);

        InputAction &cursorAction = actionMap->addAction("CursorState", AN::kInputControlButton);
        cursorAction.addBinding(kInputKey_K);

        InputAction &fullScreenAction = actionMap->addAction("ToggleFullScreen", AN::kInputControlButton);
        fullScreenAction.addBinding(kInputKey_F);

        return actionMap;
    }

    virtual void gameSetup(AN::Game &game) override {
        GetConfiguration().setObject("anti-aliasing", (AntiAliasingMethod)kAntiAliasingNone);
        GetConfiguration().setObject("msaa-samples", 1U);

        int refleshRate = AN::GetScreen().getRefreshRate() * 2;
        game.setMaxFrameRate(refleshRate);

        //        Cursor::setState(AN::kCursorStateDisabled);

        Actor *camera = NewObject<Actor>();
        camera->init("MainCamera");
        Camera *cameraCom = camera->addComponent<Camera>();
        cameraCom->setViewportRatio((float)size.width / size.height);

        InputActionMap *actionMap = createInputActionMap();
        InputComponent *inputComponent = camera->addComponent<InputComponent>();
        inputComponent->setActionMap(actionMap);

        camera->addComponent<IMGUIDemo>();
    }
};

int main(int argc, const char *argv[]) {

    AN::Application &app = AN::Application::GetSharedApplication();
    AppDelegate *appDelegate = new AppDelegate();
    app.setDelegate(appDelegate);
    appDelegate->release();

    try {
        app.run();

    } catch (const std::exception &exception) {

#ifdef _WIN32
        MessageBoxA(nullptr, exception.what(), "Exception", MB_OK|MB_ICONERROR);
#endif

    }
    return 0;
}