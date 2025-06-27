import FormContext from '@/context/FormContext'
import React, { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { toast } from 'sonner';
import { cn } from '@/lib/utils';

type ActionFunction = (formData: FormData) => Promise<any>;

const FormContainer = ({action, children, className} : {action: ActionFunction, children: React.ReactNode, className?: string}) => {
    const [message, setMessage] = useState<string>('')
    const [loading, setLoading] = useState<boolean>(false)

    const mutation = useMutation({
        mutationFn: async (formData: FormData) => {
          return await action(formData);
        },
        onMutate: () => {
          setLoading(true);
        },
        onSuccess: ({message, deskripsi}) => {
          setMessage(message || '');
          setLoading(false);
          toast(message, {description: deskripsi})
        },
        onError: (error: any) => {
          setMessage(error.message || 'Something went wrong');
          setLoading(false);
          toast("Terjadi Kesalahan", {description: error.response.data.error})
        },
      });
      
      const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        const formData = new FormData(e.currentTarget);
        mutation.mutate(formData);
      };

    return (
        <FormContext.Provider value={{
            isLoading: loading
        }}>
            <form encType='multipart/form-data' className={cn('', className)} onSubmit={handleSubmit}>
                {children}
            </form>
        </FormContext.Provider>
    )
}

export default FormContainer