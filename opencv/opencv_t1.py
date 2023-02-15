import cv2

from opencv.template_matching import TemplateMatching

# 读取目标图片
target = cv2.imread("t4.png")
# 读取模板图片
template = cv2.imread("t1.png")

result = TemplateMatching(target, template, 0.5).find_best_result()

print(result)

# print(result['result'])
# print(result['confidence'])
# print(result['rectangle'])
# cv2.rectangle(template, result['rectangle'][1], result['rectangle'][3], (0, 0, 255), 2)
#
# cv2.imwrite("t3.jpg", template)
# cv2.imshow("template", template)
# cv2.waitKey()
