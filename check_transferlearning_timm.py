import timm
import sys

if __name__ == '__main__':
    select_num = ['0', '1', '2']
    
    print('transfer learning을 위한 파라미터 체크 프로그램')
    print('원하시는 메뉴를 숫자로 선택하세요.')
    print('0 : 프로그램 종료')
    print('1 : timm내 모델 찾기')
    print('2 : 모델의 세부 파라미터 파악')
    print('--------------------')
    
    num = input('이곳에 다음의 숫자를 입력해 주세요 : ')
    
    while num not in select_num:
        num = input('다시 입력해주세요. : ')
    
    if num == '0' :
        print('프로그램을 종료합니다.')
        sys.exit(0)
    
    elif num == '1':
        # 1번 : timm내 모델 찾기
        print('찾고 싶은 모델을 입력하세요.')
        print('예 : efficientnet 관련 모델을 찾고 싶은 경우 -> *efficientnet* 입력')
        search_model = input('이곳에 입력하세요 : ')

        try:
            model_list = timm.list_models(search_model)
            print('--------------------')
            print(f'{search_model}에 대한 검색결과는 다음과 같습니다.')
            print(f'검색 결과, 총 {len(model_list)}개의 리스트를 찾았습니다.')
            print(model_list)
            print('--------------------')
        except Exception as e:
            print(f'error 발생 : {e}')
            print('프로그램을 종료합니다. ')
    
    elif num == '2':
        # 2번 : 찾은 모델 기반 resize, mean, std 체크
        model_name = input('확인하고 싶은 모델의 이름을 정확하게 알려주세요. : ')
        print('모델 이름 확인 완료. 모델을 불러오는 중입니다.')
        model = timm.create_model(model_name, pretrained=False)
        print('모델 불러오기 완료. 모델의 상세 정보를 파악합니다.')
        
        pretrained_cfg = model.pretrained_cfg
        
        input_size = pretrained_cfg['input_size']
        mean = pretrained_cfg['mean']
        std = pretrained_cfg['std']
        
        print('--------------------')
        print('모델 정보 확인 완료!')
        print(f'{model_name}의 정보는 아래와 같습니다.')
        print(f"input size (H x W) : {input_size[1]} x {input_size[2]}")
        print(f"mean : {mean}")
        print(f"std : {std}")
        print('--------------------')
    
                              